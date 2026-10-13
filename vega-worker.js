// Vega 0.8B decision runtime: onnxruntime-web (wasm) + tokenizers.js, text states only.
// Prompt, pooling and calibration follow the base model's vega_common / vega_api (views=1),
// as documented on huggingface.co/nvkudva/vega-onnx-q8; token ids are checked against golden ids.

import { pyJsonDumps } from "./py-json.js";

// ─── Constants ───
const ORT_VERSION = "1.30.0";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const TOKENIZERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const MODEL_REVISION = "017c096d369ac698087f8c35a6ad8c867a7fe0f4";
const BASE = `https://huggingface.co/nvkudva/vega-onnx-q8/resolve/${MODEL_REVISION}/v2`;
const CACHE_NAME = `vega-weights-${MODEL_REVISION.slice(0, 8)}`;
const WEIGHT_FILES = ["embed_q8.onnx", "embed_q8.onnx.data", "decoder_q8.onnx", "decoder_q8.onnx.data", "engine.onnx"];
const TEXT_FILES = ["vega_web.json", "tokenizer.json", "tokenizer_config.json"];
const HIDDEN = 1024;
const QTYPES = { choice: 0, score: 1, boolean: 2 };
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 32;
const MAX_QUESTIONS = 8;
// The model reads 73k tokens, but every question re-reads the state on one wasm CPU.
const BROWSER_MAX_TOKENS = 4096;
const ALPHA = "0.1";
const TEMPLATE = { head: "Observation:\n", q: "\n\nMeasurement ({t}): {ins}\nPossible outcomes:", opt: "\n* {o}", end: "\nOutcome:" };
const QUERY_FIRST = "Question to keep in mind while reading ({t}): {ins}\n\n";
// Golden case multi_q_mixed from the reference parity set: one dict state, all three types.
const SELF_CHECK = {
  state: { channel: "email", from: "ops@vendor.example", subject: "Invoice 88123 overdue", body: "Payment terms were net 30. We are now at day 47." },
  questions: {
    a: { type: "boolean", instructions: "Is this about money?", criteria: null },
    b: { type: "choice", instructions: "Route it.", criteria: { finance: "invoices", legal: "contracts", support: "everything else" } },
    c: { type: "score", instructions: "How urgent?", criteria: ["low", "medium", "high"] },
  },
  probs: {
    a: [0.011699715031814275, 0.9883002849681857],
    b: [0.9085255158608415, 0.06007649919049143, 0.03139798494866709],
    c: [0.05379994833586495, 0.45333817115432923, 0.4928618805098059],
  },
  // shared prefix ids are identical across the three prompts; the tails differ
  prefixA: [14162, 310, 2426, 303, 3838, 1345, 5122, 318, 5925, 1590, 2091, 411, 883, 3117, 30, 271, 36514, 362, 25, 198],
  idsLength: { a: 103, b: 99, c: 104 },
  // 8-bit vs fp32 reference: the published max |dp| over 30 cases is 0.0503
  tolerance: 0.06,
};

// ─── State ───
let ort;
let tokenizer;
let cfg;
let sessions;
let pastFeeds;
let busy = Promise.resolve();

// ─── Helpers ───
const send = (type, data = {}) => self.postMessage({ type, ...data });
const encode = (text) => Array.from(tokenizer.encode(text, { add_special_tokens: false }).ids);

async function openCache() {
  try { return await caches.open(CACHE_NAME); } catch { return null; }
}

async function cachedFiles() {
  const cache = await openCache();
  if (!cache) return [];
  const hits = await Promise.all(WEIGHT_FILES.map(async (file) => ((await cache.match(`${BASE}/${file}`)) ? file : null)));
  return hits.filter(Boolean);
}

// Fetch with byte progress, kept in the Cache API so a reload does not download 840 MB again.
async function fetchBytes(file, cache) {
  const url = `${BASE}/${file}`;
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
  const chunks = [];
  let loaded = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (loaded - lastReport > 4_000_000 || loaded === total) {
      lastReport = loaded;
      send("progress", { file, loaded, total: total || loaded, cached: false });
    }
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  send("progress", { file, loaded, total: loaded, cached: false });
  try { await cache?.put(url, new Response(bytes)); } catch { /* quota: run uncached */ }
  return bytes;
}

const fetchJson = async (file, cache) => JSON.parse(new TextDecoder().decode(await fetchBytes(file, cache)));

const critText = (value) => (typeof value === "string" ? value : pyJsonDumps(value, false));
const isSet = (value) => value !== null && value !== undefined && value !== "";

// Request question -> internal form; list criteria of a choice become {label: null}.
function toInternal(question) {
  let crit = question.criteria ?? null;
  if (question.type === "choice" && Array.isArray(crit)) crit = Object.fromEntries(crit.map((c) => [String(c), null]));
  const ins = typeof question.instructions === "string" ? question.instructions : pyJsonDumps(question.instructions, true);
  return { t: question.type, ins, crit };
}

// Option texts in label order. Boolean is always [no, yes].
function optionTexts(q) {
  if (q.t === "choice") return Object.entries(q.crit).map(([k, v]) => (isSet(v) ? `${k} - ${critText(v)}` : k));
  if (q.t === "score") return q.crit.map((c, i) => `level ${i} - ${critText(c)}`);
  const crit = q.crit ?? {};
  return [
    "no - " + (isSet(crit.false) ? critText(crit.false) : "the statement does not hold"),
    "yes - " + (isSet(crit.true) ? critText(crit.true) : "the statement holds"),
  ];
}

function labelsFor(q) {
  if (q.t === "choice") return Object.keys(q.crit);
  if (q.t === "score") return q.crit.map((_, i) => i);
  return [false, true];
}

// State first so its span never sees the question, unless the state is short enough for the
// query-first preamble. Qwen adds no BOS.
function buildPrompt(stateText, q, insMaxTokens = 256) {
  const maxLen = Math.min(cfg.max_len, BROWSER_MAX_TOKENS);
  const fill = (s) => s.replace("{t}", () => q.t).replace("{ins}", () => String(q.ins));
  const head = encode(TEMPLATE.head);
  let qids = encode(fill(TEMPLATE.q)).slice(0, insMaxTokens);
  let oids = optionTexts(q).map((o) => encode(TEMPLATE.opt.replace("{o}", () => o)).slice(0, cfg.opt_max_tokens));
  const end = encode(TEMPLATE.end);
  const total = (lists) => lists.reduce((sum, list) => sum + list.length, 0);
  let fixed = head.length + qids.length + total(oids) + end.length;
  if (fixed > cfg.max_len - 16) {
    const per = Math.max(3, Math.floor((cfg.max_len - 16 - head.length - end.length - Math.min(qids.length, 64)) / Math.max(1, oids.length)));
    oids = oids.map((o) => o.slice(0, per));
    qids = qids.slice(0, Math.max(8, cfg.max_len - 16 - head.length - end.length - total(oids)));
    fixed = head.length + qids.length + total(oids) + end.length;
  }
  const stateIds = encode(stateText);
  let pre = [];
  if (cfg.query_first_max_state && stateIds.length < cfg.query_first_max_state) {
    pre = encode(fill(QUERY_FIRST)).slice(0, Math.min(128, insMaxTokens));
    fixed += pre.length;
  }
  const room = Math.max(0, maxLen - fixed);
  // refuse instead of truncating: a decision on a cut-off record is worse than none
  if (stateIds.length > room) throw new Error(`The state is ${stateIds.length} tokens; only ${room} fit in this page's ${maxLen}-token browser limit.`);
  const ids = [...pre, ...head];
  const s0 = ids.length;
  ids.push(...stateIds);
  const state = stateIds.length ? [s0, ids.length] : [0, s0];
  const q0 = ids.length;
  ids.push(...qids);
  const question = [q0, ids.length];
  const options = oids.map((o) => {
    const a = ids.length;
    ids.push(...o);
    return [a, ids.length];
  });
  ids.push(...end);
  return { ids, spans: { state, question, options } };
}

// Span means per feature layer (state, question, final token), option span means at layer 19,
// and up to 16 equal chunk means of a long state for the engine's memory reader.
function pool(h13, h19, prompt) {
  const n = prompt.ids.length;
  const D = HIDDEN;
  const mean = (h, a, e, out, offset) => {
    if (e <= a) return;
    for (let t = a; t < e; t++) for (let j = 0; j < D; j++) out[offset + j] += h[t * D + j];
    for (let j = 0; j < D; j++) out[offset + j] /= e - a;
  };
  const { state, question, options } = prompt.spans;
  const feats = new Float32Array(2 * 3 * D);
  [h13, h19].forEach((h, layer) => {
    mean(h, state[0], state[1], feats, (layer * 3) * D);
    mean(h, question[0], question[1], feats, (layer * 3 + 1) * D);
    feats.set(h.subarray((n - 1) * D, n * D), (layer * 3 + 2) * D);
  });
  const opts = new Float32Array(options.length * D);
  options.forEach(([a, e], k) => mean(h19, a, e, opts, k * D));
  const C = cfg.memory_chunks;
  const chunks = new Float32Array(C * D);
  const chunkMask = new Uint8Array(C);
  const m = state[1] - state[0];
  if (m >= Math.max(cfg.memory_min_tokens, 2)) {
    const k = Math.max(2, Math.min(C, Math.floor(m / Math.max(1, cfg.memory_chunk_tokens))));
    let a = 0;
    for (let c = 0; c < k; c++) {
      let e = a;
      while (e < m && Math.floor((e * k) / m) === c) e++;
      mean(h19, state[0] + a, state[0] + e, chunks, c * D);
      a = e;
    }
    chunkMask.fill(1, 0, k);
  }
  return { feats, opts, chunks, chunkMask };
}

// Thermal calibration: log tau = b + w . [log1p kinetic, log1p nearest, steps/n_steps, log k].
function calibrate(theta, qtype, engine) {
  const k = engine.logits.length;
  const phys = [Math.log1p(engine.kinetic), Math.log1p(engine.nearest), engine.steps / Math.max(1, cfg.n_steps), Math.log(Math.max(k, 2))];
  const th = theta[qtype];
  const logTau = th[0] + phys.reduce((sum, v, i) => sum + th[i + 1] * v, 0);
  const tau = Math.exp(Math.min(Math.max(logTau, Math.log(0.2)), Math.log(20)));
  const z = engine.logits.map((v) => v / tau);
  const top = Math.max(...z);
  const ex = z.map((v) => Math.exp(v - top));
  const sum = ex.reduce((a, b) => a + b, 0);
  return { probs: ex.map((v) => v / sum), tau };
}

function answerFor(q, engine) {
  const threshold = cfg.adapters[0]?.gate_threshold ?? 0.5;
  const gate = engine.gates.length ? Math.max(...engine.gates) : 0;
  const route = engine.gates.length && gate >= threshold ? engine.gates.indexOf(gate) : -1;
  const cal = route < 0 ? cfg.calibration : cfg.adapters[route].calibration;
  const { probs, tau } = calibrate(cal.theta, QTYPES[q.t], engine);
  const labels = labelsFor(q);
  const confidence = Math.max(...probs);
  const best = probs.indexOf(confidence);
  const conformal = cal.conformal[q.t]?.[ALPHA];
  // LAC prediction set; the top option always stays in so the set is never empty
  const predictionSet = conformal === undefined ? null : labels.filter((_, i) => probs[i] >= Math.max(1 - conformal, 1e-12) || i === best);
  const risk = cfg.calibration.default_risk ?? "0.1";
  const answer = {
    type: q.t,
    labels: labels.map(String),
    options: optionTexts(q),
    probs,
    confidence,
    prediction_set: predictionSet,
    abstain: confidence < (cal.abstain_threshold[risk] ?? 0),
    unbound: engine.nearest > (cal.unbound_threshold ?? Infinity),
    settle_steps: engine.steps,
    energy: -Math.max(...engine.logits),
    temperature: tau,
    adapter: route >= 0 ? cfg.adapters[route].name : null,
    gate,
  };
  if (q.t === "choice") answer.choice = labels[best];
  else if (q.t === "score") {
    answer.score = best;
    answer.expected_score = probs.reduce((sum, p, i) => sum + i * p, 0);
  } else answer.boolean = probs[1];
  return answer;
}

// Zero conv / recurrent / KV state for every past_* input: a first read of the sequence.
function emptyPast(session) {
  const feeds = {};
  for (const meta of session.inputMetadata) {
    if (!meta.isTensor || !meta.name.startsWith("past_")) continue;
    const shape = meta.shape.map((d) => (typeof d === "number" ? d : d === "batch_size" ? 1 : 0));
    feeds[meta.name] = new ort.Tensor("float32", new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  }
  return feeds;
}

function validateRequest(state, questions) {
  if (typeof state !== "string" && (typeof state !== "object" || state === null)) throw new Error("The state must be text or a JSON object/array.");
  if (!questions || typeof questions !== "object" || Array.isArray(questions)) throw new Error("Questions must be a JSON object of {id: {type, instructions, criteria}}.");
  const entries = Object.entries(questions);
  if (!entries.length) throw new Error("Add at least one question.");
  if (entries.length > MAX_QUESTIONS) throw new Error(`At most ${MAX_QUESTIONS} questions per run.`);
  for (const [id, q] of entries) {
    if (!q || typeof q !== "object") throw new Error(`Question "${id}" must be an object.`);
    if (!(q.type in QTYPES)) throw new Error(`Question "${id}": type must be boolean, choice or score.`);
    if (q.instructions === undefined || q.instructions === "") throw new Error(`Question "${id}" needs instructions.`);
    const crit = q.criteria ?? null;
    if (q.type === "boolean" && crit !== null && (typeof crit !== "object" || Array.isArray(crit))) throw new Error(`Question "${id}": boolean criteria must be null or {"false": ..., "true": ...}.`);
    if (q.type === "score" && !Array.isArray(crit)) throw new Error(`Question "${id}": score criteria must be a list of ordered levels.`);
    if (q.type === "choice" && (typeof crit !== "object" || crit === null)) throw new Error(`Question "${id}": choice criteria must be a list or an object of options.`);
    const count = q.type === "boolean" ? 2 : Array.isArray(crit) ? crit.length : Object.keys(crit).length;
    if (count < MIN_OPTIONS || count > MAX_OPTIONS) throw new Error(`Question "${id}": use ${MIN_OPTIONS} to ${MAX_OPTIONS} options.`);
    if (q.type === "choice" && Array.isArray(crit) && new Set(crit.map(String)).size !== crit.length) throw new Error(`Question "${id}": option names must be unique.`);
  }
}

// One embed + decoder + engine pass per question.
async function decide(state, questions, onAnswer) {
  validateRequest(state, questions);
  const t0 = performance.now();
  const stateText = typeof state === "string" ? state : pyJsonDumps(state, false);
  const answers = {};
  let tokens = 0;
  for (const [qid, question] of Object.entries(questions)) {
    const q0 = performance.now();
    const q = toInternal(question);
    const prompt = buildPrompt(stateText, q);
    const L = prompt.ids.length;
    const { inputs_embeds } = await sessions.embed.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(prompt.ids, BigInt), [1, L]),
    });
    const positions = new BigInt64Array(3 * L);
    for (let i = 0; i < L; i++) positions[i] = positions[L + i] = positions[2 * L + i] = BigInt(i);
    const { h13, h19 } = await sessions.decoder.run({
      ...pastFeeds,
      inputs_embeds: new ort.Tensor("float32", inputs_embeds.data, [1, L, HIDDEN]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]),
      position_ids: new ort.Tensor("int64", positions, [3, 1, L]),
    });
    const pooled = pool(h13.data, h19.data, prompt);
    const k = prompt.spans.options.length;
    const C = cfg.memory_chunks;
    const out = await sessions.engine.run({
      feats: new ort.Tensor("float32", pooled.feats, [1, 2, 3, HIDDEN]),
      opts: new ort.Tensor("float32", pooled.opts, [1, k, 1, HIDDEN]),
      opt_mask: new ort.Tensor("bool", new Uint8Array(k).fill(1), [1, k]),
      qtype: new ort.Tensor("int64", BigInt64Array.from([BigInt(QTYPES[q.t])]), [1]),
      chunks: new ort.Tensor("float32", pooled.chunks, [1, C, HIDDEN]),
      chunk_mask: new ort.Tensor("bool", pooled.chunkMask, [1, C]),
    });
    const answer = answerFor(q, {
      logits: Array.from(out.logits.data).slice(0, k),
      kinetic: out.kinetic.data[0],
      nearest: out.nearest.data[0],
      steps: Number(out.steps_used.data[0]),
      gates: Array.from(out.gates.data),
    });
    answer.tokens = L;
    answer.ms = performance.now() - q0;
    answer.ids = prompt.ids;
    answers[qid] = answer;
    tokens += L;
    onAnswer?.(qid, answer);
  }
  return { answers, tokens, ms: performance.now() - t0 };
}

// Validated questions with the exact option texts Vega reads; needs no weights, so it works before load.
function plan(state, questions) {
  validateRequest(state, questions);
  return Object.fromEntries(Object.entries(questions).map(([qid, question]) => {
    const q = toInternal(question);
    return [qid, { type: q.t, instructions: q.ins, options: optionTexts(q), labels: labelsFor(q).map(String) }];
  }));
}

async function selfCheck() {
  const result = await decide(SELF_CHECK.state, SELF_CHECK.questions);
  const problems = [];
  let maxDelta = 0;
  for (const [qid, expected] of Object.entries(SELF_CHECK.probs)) {
    const answer = result.answers[qid];
    if (answer.ids.length !== SELF_CHECK.idsLength[qid]) problems.push(`${qid}: ${answer.ids.length} tokens, expected ${SELF_CHECK.idsLength[qid]}`);
    const delta = Math.max(...expected.map((p, i) => Math.abs(p - answer.probs[i])));
    maxDelta = Math.max(maxDelta, delta);
    if (delta > SELF_CHECK.tolerance) problems.push(`${qid}: |dp| ${delta.toFixed(3)}`);
    if (expected.indexOf(Math.max(...expected)) !== answer.probs.indexOf(answer.confidence)) problems.push(`${qid}: different top option`);
  }
  const prefix = result.answers.a.ids.slice(0, SELF_CHECK.prefixA.length);
  if (prefix.some((id, i) => id !== SELF_CHECK.prefixA[i])) problems.push("tokenizer ids differ from the reference");
  return { ok: !problems.length, problems, maxDelta, ms: result.ms };
}

async function load() {
  const t0 = performance.now();
  send("phase", { phase: "runtime", message: "Loading onnxruntime-web and the tokenizer…" });
  ort = await import(`${ORT_DIST}ort.wasm.min.mjs`);
  const { Tokenizer } = await import(TOKENIZERS_URL);
  const threads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = threads;
  const cache = await openCache();
  const [config, tokenizerJson, tokenizerConfig] = await Promise.all(TEXT_FILES.map((file) => fetchJson(file, cache)));
  cfg = config;
  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  send("phase", { phase: "download", message: "Fetching Vega weights (840 MB on the first visit)…" });
  const make = async (name, external) => {
    const graph = await fetchBytes(`${name}.onnx`, cache);
    const data = external ? await fetchBytes(`${name}.onnx.data`, cache) : null;
    send("phase", { phase: "session", message: `Preparing ${name}…` });
    return ort.InferenceSession.create(graph, {
      executionProviders: ["wasm"],
      externalData: data ? [{ data, path: `${name}.onnx.data` }] : undefined,
    });
  };
  const embed = await make("embed_q8", true);
  const decoder = await make("decoder_q8", true);
  const engine = await make("engine", false);
  sessions = { embed, decoder, engine };
  pastFeeds = emptyPast(decoder);
  const loadMs = performance.now() - t0;
  send("loaded", { loadMs, threads, isolated: self.crossOriginIsolated });
  send("phase", { phase: "check", message: "Self-check against the reference outputs…" });
  const check = await selfCheck();
  send("selfcheck", check);
  if (!check.ok) throw new Error(`Self-check failed (${check.problems.join("; ")}). This browser computes Vega differently from the reference, so answers are not shown.`);
  send("ready", { threads, maxTokens: BROWSER_MAX_TOKENS, maxOptions: MAX_OPTIONS, maxQuestions: MAX_QUESTIONS });
}

// ─── Listeners ───
self.addEventListener("message", ({ data }) => {
  // planning needs no sessions, so it answers even while the weights are still loading
  if (data.type === "plan") {
    try { send("plan", { runId: data.runId, questions: plan(data.state, data.questions) }); }
    catch (error) { send("error", { runId: data.runId, during: "plan", message: error?.message || String(error) }); }
    return;
  }
  // one job at a time: concurrent runs on the same sessions would interleave
  busy = busy.then(async () => {
    try {
      if (data.type === "probe") send("probe", { cached: await cachedFiles(), total: WEIGHT_FILES.length });
      else if (data.type === "load") await load();
      else if (data.type === "decide") {
        const result = await decide(data.state, data.questions, (qid, answer) => send("answer", { runId: data.runId, qid, answer }));
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
