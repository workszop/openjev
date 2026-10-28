// d1-2B decision runtime: onnxruntime-web (WebGPU, fp32 activations) + tokenizers.js, text states only.
// Prompt, option codes, question packing and readout follow the checkpoint's own prompt.py, attention.py (pack_set)
// and runner.py, as published in huggingface.co/LiquidAI/d1-2B-ONNX; every question about one state is read in
// one forward pass. The fp16 graphs miscompute on some GPUs (seen on Intel Xe-LPG), so this uses decoder_q4
// (int4/int8 weights, fp32 activations); a self-check against the reference answers gates every load.

import { pyJsonDumps } from "./py-json.js";

// ─── Constants ───
const ORT_VERSION = "1.30.0";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const TOKENIZERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const MODEL_REPO = "LiquidAI/d1-2B-ONNX";
const MODEL_REVISION = "7f5c6801cba6164dda9daae925178f89717b960f";
const DEFAULT_BASE = `https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}`;
const CACHE_NAME = `d1-weights-${MODEL_REVISION.slice(0, 8)}`;
const WEIGHT_FILES = ["onnx/decoder_q4.onnx", "onnx/decoder_q4.onnx_data"];
const TEXT_FILES = ["tokenizer.json", "tokenizer_config.json", "calibration.json"];
const HIDDEN = 2048;
// Page question type -> d1 type. A boolean is d1's noul, read as [yes, no] and answered as [no, yes].
const D1_TYPE = { boolean: "noul", choice: "choice", score: "score" };
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 32;
const MAX_LEVELS = 10;
// The checkpoint's trained set size (d1_set_size): this many questions share one pass after their state.
const MAX_QUESTIONS = 16;
// The model card's WebGPU limit for decoder_q4: at most 4,096 tokens in a packed row.
const BROWSER_MAX_TOKENS = 4096;
const BOS = "<|startoftext|>";
const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";
const LEAD = "<think></think>"; // lfm3's non-thinking assistant lead (lfm3.py LEADS)
const QUESTION_HEADER = "QUESTION:\n";
const BREAK = "​"; // after a special token's first character: reads the same, is no longer the token
const LETTERS = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"];
const YES_FORMS = ["yes", "Yes", "YES"];
const NO_FORMS = ["no", "No", "NO"];
// prompt.py _FALLBACK_POOL: codes tried in order when an option's own code is not one token
const FALLBACK_POOL = [
  ...LETTERS,
  ...Array.from({ length: 100 }, (_, i) => String(i).padStart(2, "0")),
  ..."abcdefghijklmnopqrstuvwxyz",
  ...Array.from({ length: 200 }, (_, i) => `#${i}`),
  ...LETTERS.flatMap((a) => LETTERS.map((b) => a + b)),
];
// The model card's quick-start ticket, without its multiple choice. Reference: onnxruntime CPU on the same
// decoder_q4 graph, prompts built by the checkpoint's prompt.py.
const SELF_CHECK = {
  state: "I was charged twice and my parcel never arrived. Please refund one charge.",
  questions: {
    refund: { type: "boolean", instructions: "Is the customer asking for a refund?", criteria: null },
    team: { type: "choice", instructions: "Which team should handle this?", criteria: { billing: "Charges, refunds, invoices", shipping: "Delivery and parcels", technical: "App or site faults" } },
    urgency: { type: "score", instructions: "How urgent is this ticket?", criteria: ["low", "medium", "high"] },
  },
  probs: {
    refund: [0.0023676028640132797, 0.9976323971359867],
    team: [0.748213362749075, 0.20987701333538836, 0.04190962391553666],
    urgency: [0.19093452072572803, 0.6425246276295208, 0.16654085164475124],
  },
  ids: [124894, 124899, 5922, 207, 49, 452, 14917, 10625, 309, 1226, 60525, 2966, 10922, 22, 13330, 52966, 734, 7285, 64673, 63509, 833, 4439, 278, 13107],
  idsLength: 123,
  tolerance: 0.05,
};

// ─── State ───
let ort;
let tokenizer;
let session;
let specials = null; // RegExp over the tokenizer's special tokens
let temperature = 1;
let busy = Promise.resolve();
const singleIds = new Map(); // text -> its one token id, or null

// ─── Helpers ───
const send = (type, data = {}) => self.postMessage({ type, ...data });
const encode = (text) => Array.from(tokenizer.encode(text, { add_special_tokens: false }).ids);
const critText = (value) => (typeof value === "string" ? value : pyJsonDumps(value, false));
const isSet = (value) => value !== null && value !== undefined && value !== "";
const pyStr = (value) => (value === null || value === undefined ? "None" : typeof value === "boolean" ? (value ? "True" : "False") : String(value));

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function openCache() {
  try { return await caches.open(CACHE_NAME); } catch { return null; }
}

async function cachedFiles(base) {
  const cache = await openCache();
  if (!cache) return [];
  const hits = await Promise.all(WEIGHT_FILES.map(async (file) => ((await cache.match(`${base}/${file}`)) ? file : null)));
  return hits.filter(Boolean);
}

// Fetch with byte progress into one buffer sized from content-length (a 1.7 GB file is never held twice),
// kept in the Cache API so a reload does not download it again.
async function fetchBytes(base, file, cache) {
  const url = `${base}/${file}`;
  const hit = cache ? await cache.match(url).catch(() => null) : null;
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    send("progress", { file, loaded: bytes.length, total: bytes.length, cached: true });
    return bytes;
  }
  const response = await fetch(url, { referrerPolicy: "no-referrer" });
  if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length") || 0);
  const reader = response.body.getReader();
  let bytes = new Uint8Array(total || 1 << 20);
  let loaded = 0;
  let lastSent = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > bytes.length) { // no or wrong content-length: grow
      const grown = new Uint8Array(Math.max(bytes.length * 2, loaded + value.length));
      grown.set(bytes.subarray(0, loaded));
      bytes = grown;
    }
    bytes.set(value, loaded);
    loaded += value.length;
    if (loaded - lastSent > 4_000_000) {
      send("progress", { file, loaded, total: total || loaded, cached: false });
      lastSent = loaded;
    }
  }
  if (loaded !== bytes.length) bytes = bytes.slice(0, loaded);
  send("progress", { file, loaded, total: loaded, cached: false });
  if (cache) await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  return bytes;
}

const fetchJson = async (base, file, cache) => JSON.parse(new TextDecoder().decode(await fetchBytes(base, file, cache)));

// prompt.plain: the caller's text with every special token broken, so a state holding <|im_end|> stays text.
function plain(text) {
  if (!specials) return text;
  let out = text;
  for (;;) { // again for a match that overlapped the one broken before it
    let n = 0;
    out = out.replace(specials, (m) => { n += 1; return m[0] + BREAK + m.slice(1); });
    if (!n) return out;
  }
}

function singleId(text) {
  if (!singleIds.has(text)) {
    const ids = encode(text);
    singleIds.set(text, ids.length === 1 ? ids[0] : null);
  }
  return singleIds.get(text);
}

// prompt._ids: the texts that are one token each, each id once.
function idsOf(texts) {
  const out = [];
  for (const text of texts) {
    const id = singleId(text);
    if (id !== null && !out.includes(id)) out.push(id);
  }
  return out;
}

// prompt.option_codes, scheme 2 (d1_codes): native letters, else A..Z, then AA, AB..
function optionCodes(labels) {
  const labs = labels.map((x) => String(x).trim());
  if (labs.length && labs.every((k) => /^\p{L}$/u.test(k))) return labs;
  return [...LETTERS, ...LETTERS.flatMap((a) => LETTERS.map((b) => a + b))].slice(0, labs.length);
}

// prompt.aliases: every option's distinct single-token code, [code, token id].
function aliases(labels) {
  const used = new Set();
  const out = [];
  const take = (raw) => {
    const id = singleId(raw);
    if (id === null || used.has(id)) return false;
    out.push([raw, id]);
    used.add(id);
    return true;
  };
  for (const code of optionCodes(labels)) {
    if (!take(code) && !FALLBACK_POOL.some(take)) throw new Error(`no single-token code left for ${labels.length} options`);
  }
  return out;
}

// Page question -> d1 question: { type, instructions, criteria } as prompt.as_question reads it.
function toD1(question) {
  const type = D1_TYPE[question.type];
  let criteria = question.criteria ?? null;
  if (type === "choice" && Array.isArray(criteria)) criteria = Object.fromEntries(criteria.map((c) => [String(c), null]));
  return { type, instructions: question.instructions, criteria };
}

// prompt.question_block with option_style "desc"; codes are the choice's aliases.
function questionBlock(q, codes) {
  if (q.type === "choice") {
    const lines = Object.entries(q.criteria).map(([label, desc], i) => `${codes[i][0]} ${desc || label.replaceAll("_", " ")}`);
    return `${q.instructions}\n\nOptions:\n${lines.join("\n")}\n\nReply with the option code only.`;
  }
  if (q.type === "noul") {
    const c = q.criteria;
    const extra = c && Object.keys(c).length ? `\nYes: ${pyStr(c.true)}\nNo: ${pyStr(c.false)}` : "";
    return `${q.instructions}${extra}\n\nReply with yes or no only.`;
  }
  const legend = q.criteria.map((name, i) => `${i} ${name}`).join("\n");
  return `${q.instructions}\n\n${legend}\n\nReply with a single digit 0-${q.criteria.length - 1} only.`;
}

// prompt.readout_ids: token ids per option, max-pooled at the answer slot.
function readoutIds(q, codes) {
  if (q.type === "noul") {
    const yes = idsOf(YES_FORMS);
    const no = idsOf(NO_FORMS);
    if (!yes.length || !no.length) throw new Error("the tokenizer has no single-token yes/no");
    return [yes, no];
  }
  if (q.type === "score") {
    return q.criteria.map((_, i) => {
      const group = idsOf([String(i)]);
      if (!group.length) throw new Error(`score level ${i} is not one token`);
      return group;
    });
  }
  return codes.map(([code, id]) => [id, ...idsOf([` ${code}`]).filter((x) => x !== id)]);
}

// prompt.state_block in the default json_only style: text as it is, any other state as indented JSON.
const stateBlock = (state) => `${typeof state === "string" ? state : pyJsonDumps(state, false, 2)}\n\n`;
const prefixText = (state) => `${BOS}${IM_START}user\n${plain(stateBlock(state))}\n${QUESTION_HEADER}`;
const suffixText = (body) => `${plain(body)}${IM_END}\n${IM_START}assistant\n${LEAD}`;

// attention.pack_set: the state with its first question, tokenized as its lone prompt; each further question its own
// group, the header then its suffix. question_ids is -1 on the state and k on question k; each slot is a suffix's end.
function packSet(prefix, suffixes) {
  const ids = encode(prefix + suffixes[0]);
  const state = ids.length - encode(suffixes[0]).length;
  const questionIds = [...new Array(state).fill(-1), ...new Array(ids.length - state).fill(1)];
  const slots = [ids.length - 1];
  suffixes.slice(1).forEach((suffix, i) => {
    const own = encode(QUESTION_HEADER + suffix);
    ids.push(...own);
    questionIds.push(...new Array(own.length).fill(i + 2));
    slots.push(ids.length - 1);
  });
  return { ids, state, questionIds, slots };
}

// Option texts the page and Qwen3 read, in plan order. A boolean is always [no, yes].
function optionTexts(question) {
  const crit = question.criteria ?? null;
  if (question.type === "choice") {
    const entries = Array.isArray(crit) ? crit.map((k) => [String(k), null]) : Object.entries(crit);
    return entries.map(([k, v]) => (isSet(v) ? `${k} - ${critText(v)}` : k));
  }
  if (question.type === "score") return crit.map((c, i) => `level ${i} - ${critText(c)}`);
  const c = crit ?? {};
  return [
    "no - " + (isSet(c.false) ? critText(c.false) : "the statement does not hold"),
    "yes - " + (isSet(c.true) ? critText(c.true) : "the statement holds"),
  ];
}

function labelsFor(question) {
  if (question.type === "choice") return Array.isArray(question.criteria) ? question.criteria.map(String) : Object.keys(question.criteria);
  if (question.type === "score") return question.criteria.map((_, i) => String(i));
  return ["false", "true"];
}

function validateRequest(state, questions) {
  if (typeof state !== "string" && (typeof state !== "object" || state === null)) throw new Error("The state must be text or a JSON object/array.");
  if (!questions || typeof questions !== "object" || Array.isArray(questions)) throw new Error("Questions must be a JSON object of {id: {type, instructions, criteria}}.");
  const entries = Object.entries(questions);
  if (!entries.length) throw new Error("Add at least one question.");
  if (entries.length > MAX_QUESTIONS) throw new Error(`At most ${MAX_QUESTIONS} questions per run: d1 reads that many about one state in one pass.`);
  for (const [id, q] of entries) {
    if (!q || typeof q !== "object") throw new Error(`Question "${id}" must be an object.`);
    if (!(q.type in D1_TYPE)) throw new Error(`Question "${id}": type must be boolean, choice or score.`);
    if (typeof q.instructions !== "string" || !q.instructions.trim()) throw new Error(`Question "${id}" needs instructions as text.`);
    const crit = q.criteria ?? null;
    if (q.type === "boolean" && crit !== null && (typeof crit !== "object" || Array.isArray(crit))) throw new Error(`Question "${id}": boolean criteria must be null or {"false": ..., "true": ...}.`);
    if (q.type === "score" && !(Array.isArray(crit) && crit.every((c) => typeof c === "string"))) throw new Error(`Question "${id}": score criteria must be a list of ordered levels, as text.`);
    if (q.type === "choice" && (typeof crit !== "object" || crit === null)) throw new Error(`Question "${id}": choice criteria must be a list or an object of options.`);
    if (q.type === "choice" && !Array.isArray(crit) && Object.values(crit).some((v) => v !== null && typeof v !== "string")) throw new Error(`Question "${id}": each choice option's description must be text or null.`);
    const count = q.type === "boolean" ? 2 : Array.isArray(crit) ? crit.length : Object.keys(crit).length;
    const most = q.type === "score" ? MAX_LEVELS : MAX_OPTIONS;
    if (count < MIN_OPTIONS || count > most) throw new Error(`Question "${id}": use ${MIN_OPTIONS} to ${most} ${q.type === "score" ? "levels" : "options"}.`);
    if (q.type === "choice" && Array.isArray(crit) && new Set(crit.map(String)).size !== crit.length) throw new Error(`Question "${id}": option names must be unique.`);
  }
}

// prompt.readout through the served temperature: each option's best form, divided by T, as a softmax.
function readout(row, groups) {
  const vocab = row.length;
  return softmax(groups.map((group) => Math.max(...group.map((id) => (id < vocab ? row[id] : -Infinity))) / temperature));
}

// api.score_level, median readout: the first level whose cumulative probability reaches one half.
function medianLevel(probs) {
  let mass = 0;
  for (let i = 0; i < probs.length; i++) {
    mass += probs[i];
    if (mass >= 0.5) return i;
  }
  return probs.length - 1;
}

// Every question about the state in one forward pass of one packed row.
async function decide(state, questions, onAnswer) {
  validateRequest(state, questions);
  const t0 = performance.now();
  const entries = Object.entries(questions);
  const asked = entries.map(([qid, question]) => {
    const q = toD1(question);
    const codes = q.type === "choice" ? aliases(Object.keys(q.criteria)) : [];
    return { qid, question, q, groups: readoutIds(q, codes), suffix: suffixText(questionBlock(q, codes)) };
  });
  const row = packSet(prefixText(state), asked.map((a) => a.suffix));
  const T = row.ids.length;
  if (T > BROWSER_MAX_TOKENS) throw new Error(`The state and questions are ${T} tokens; d1 on WebGPU reads up to ${BROWSER_MAX_TOKENS} in one pass. Shorten the state or ask fewer questions.`);
  const int64 = (values, dims) => new ort.Tensor("int64", BigInt64Array.from(values, BigInt), dims);
  const feeds = {
    input_ids: int64(row.ids, [1, T]),
    attention_mask: int64(new Array(T).fill(1), [1, T]),
    position_ids: int64([...Array(T).keys()], [1, T]),
    image_features: new ort.Tensor("float32", new Float32Array(0), [0, HIDDEN]),
    video_features: new ort.Tensor("float32", new Float32Array(0), [0, HIDDEN]),
    question_ids: int64(row.questionIds, [1, T]),
    slot_positions: int64(row.slots, [row.slots.length]),
  };
  const { logits } = await session.run(feeds, ["logits"]);
  const vocab = logits.dims[1];
  const passMs = performance.now() - t0;
  const answers = {};
  asked.forEach(({ qid, question, q, groups }, k) => {
    const raw = readout(logits.data.subarray(k * vocab, (k + 1) * vocab), groups);
    const probs = q.type === "noul" ? [raw[1], raw[0]] : raw; // noul reads [yes, no]
    const confidence = Math.max(...probs);
    const answer = { type: question.type, labels: labelsFor(question), options: optionTexts(question), probs, confidence, tokens: T, ms: passMs };
    if (question.type === "choice") answer.choice = answer.labels[probs.indexOf(confidence)];
    else if (question.type === "score") answer.level = medianLevel(probs);
    else answer.boolean = probs[1];
    answers[qid] = answer;
    onAnswer?.(qid, answer);
  });
  logits.dispose?.();
  return { answers, tokens: T, ids: row.ids, ms: performance.now() - t0 };
}

// Validated questions with the option texts the page shows; needs no weights, so it works before load.
function plan(state, questions) {
  validateRequest(state, questions);
  return Object.fromEntries(Object.entries(questions).map(([qid, question]) => [qid, {
    type: question.type, instructions: question.instructions, options: optionTexts(question), labels: labelsFor(question),
  }]));
}

async function selfCheck() {
  const result = await decide(SELF_CHECK.state, SELF_CHECK.questions);
  const problems = [];
  let maxDelta = 0;
  if (result.tokens !== SELF_CHECK.idsLength) problems.push(`${result.tokens} tokens, expected ${SELF_CHECK.idsLength}`);
  if (SELF_CHECK.ids.some((id, i) => result.ids[i] !== id)) problems.push("tokenizer ids differ from the reference");
  for (const [qid, expected] of Object.entries(SELF_CHECK.probs)) {
    const { probs } = result.answers[qid];
    const delta = Math.max(...expected.map((p, i) => Math.abs(p - probs[i])));
    maxDelta = Math.max(maxDelta, delta);
    if (delta > SELF_CHECK.tolerance) problems.push(`${qid}: |dp| ${delta.toFixed(3)}`);
    if (expected.indexOf(Math.max(...expected)) !== probs.indexOf(Math.max(...probs))) problems.push(`${qid}: different top option`);
  }
  return { ok: !problems.length, problems, maxDelta, ms: result.ms, count: Object.keys(SELF_CHECK.probs).length };
}

async function load(base) {
  const t0 = performance.now();
  if (!navigator.gpu || !(await navigator.gpu.requestAdapter().catch(() => null))) throw new Error("d1-2B needs WebGPU, which this browser does not offer.");
  send("phase", { phase: "runtime", message: "Loading onnxruntime-web (WebGPU) and the tokenizer…" });
  ort = await import(`${ORT_DIST}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_DIST;
  const { Tokenizer } = await import(TOKENIZERS_URL);
  const cache = await openCache();
  const [tokenizerJson, tokenizerConfig, calibration] = await Promise.all(TEXT_FILES.map((file) => fetchJson(base, file, cache)));
  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  const names = tokenizerJson.added_tokens.filter((t) => t.special && t.content.length > 1).map((t) => t.content).sort((a, b) => b.length - a.length);
  specials = names.length ? new RegExp(names.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g") : null;
  singleIds.clear();
  // calibration.json: no per-type temperatures in this release, so its default applies to every option answer
  temperature = calibration.default ?? 1;
  send("phase", { phase: "download", message: "Fetching d1-2B weights (1.7 GB on the first visit)…" });
  const graph = await fetchBytes(base, WEIGHT_FILES[0], cache);
  const data = await fetchBytes(base, WEIGHT_FILES[1], cache);
  send("phase", { phase: "session", message: "Compiling the graph for WebGPU…" });
  session = await ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"],
    externalData: [{ data, path: "decoder_q4.onnx_data" }],
  });
  send("loaded", { loadMs: performance.now() - t0 });
  send("phase", { phase: "check", message: "Self-check against the reference answers…" });
  const check = await selfCheck();
  send("selfcheck", check);
  if (!check.ok) {
    await session.release?.().catch(() => {});
    session = null;
    throw new Error(`Self-check failed (${check.problems.join("; ")}). This GPU computes d1 differently from the reference, so answers are not shown.`);
  }
  send("ready", { maxTokens: BROWSER_MAX_TOKENS, maxOptions: MAX_OPTIONS, maxQuestions: MAX_QUESTIONS });
}

// ─── Listeners ───
self.addEventListener("message", ({ data }) => {
  const base = data.base || DEFAULT_BASE;
  // planning needs no session, so it answers even while the weights are still loading
  if (data.type === "plan") {
    try { send("plan", { runId: data.runId, questions: plan(data.state, data.questions) }); }
    catch (error) { send("error", { runId: data.runId, during: "plan", message: error?.message || String(error) }); }
    return;
  }
  // one job at a time: concurrent runs on one session would interleave
  busy = busy.then(async () => {
    try {
      if (data.type === "probe") send("probe", { cached: await cachedFiles(base), total: WEIGHT_FILES.length });
      else if (data.type === "load") await load(base);
      else if (data.type === "decide") {
        const { answers, ids, ...result } = await decide(data.state, data.questions, (qid, answer) => send("answer", { runId: data.runId, qid, answer }));
        send("decided", { runId: data.runId, ...result });
      } else if (data.type === "clear-cache") {
        await caches.delete(CACHE_NAME).catch(() => false);
        send("probe", { cached: [], total: WEIGHT_FILES.length });
      }
    } catch (error) {
      send("error", { runId: data.runId, during: data.type, message: error?.message || String(error) });
    }
  });
});

// ─── Init ───
send("worker-ready");
