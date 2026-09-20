const browserFetch = self.fetch.bind(self);
self.fetch = (input, init = {}) => browserFetch(input, { ...init, referrerPolicy: "no-referrer" });

const { Wllama, LoggerWithoutDebug } = await import("./vendor/wllama/index.js");

const MODEL = {
  name: "Qwen3 0.6B",
  url: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf",
  localFile: "qwen3-0.6b.gguf",
};

const MIN_OPTIONS = 2;
const MAX_OPTIONS = 20;
const labelsFor = (count) => Array.from({ length: count }, (_, index) => String.fromCharCode(65 + index));
let engine;

function send(type, data = {}) { self.postMessage({ type, ...data }); }
function optionBlock(options, labels = labelsFor(options.length)) {
  return options.map((option, index) => `${labels[index]}. ${option}`).join("\n");
}
function messagesFor({ state, question, options }, mode) {
  const labels = labelsFor(options.length);
  const outputInstruction = mode === "direct"
    ? `Reply with exactly one option letter from: ${labels.join(", ")}.`
    : `Estimate the probability that each allowed option is the correct decision.
Return only one JSON object mapping each option to its probability. Form every key as "<label>: <full option text>" using the allowed options above.
For example, if the unrelated options were "A. Route north" and "B. Route south", valid output would be:
{"A: Route north": 0.65, "B: Route south": 0.35}
For the actual decision, include every supplied option exactly once and in order. Each value must be a JSON number from 0 to 1, and the probabilities must sum to 1. Output JSON only, with no markdown or explanation.`;
  return [
    { role: "system", content: "Make the requested decision from the supplied state. Follow the output format exactly." },
    { role: "user", content: `State:\n${state}\n\nQuestion:\n${question}\n\nAllowed options:\n${optionBlock(options, labels)}\n\n${outputInstruction}` },
  ];
}
function softmax(values) {
  const maximum = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}
const TOP_LOGPROBS = 40;
function optionLogprobs(response, labels) {
  const entries = response.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
  const sampled = response.choices?.[0]?.message?.content?.trim();
  return labels.map((label) => {
    const ascii = label.charCodeAt(0);
    const entry = entries.find((item) => item.token === label || (item.bytes?.length === 1 && item.bytes[0] === ascii));
    if (entry && Number.isFinite(Number(entry.logprob))) return Number(entry.logprob);
    // Below the reported top-N cutoff: the option has negligible mass, so treat it as -Infinity (probability 0).
    if (label === sampled) throw new Error(`The model chose ${label} but reported no logprob for it.`);
    return -Infinity;
  });
}
function validateOptionLogprobs(values, labels) {
  if (!values || values.length !== labels.length || values.some((value) => Number.isNaN(value))) {
    throw new Error(`The model did not return valid option logits for ${labels.join(", ")}.`);
  }
  if (values.every((value) => value === -Infinity)) {
    throw new Error(`None of the option letters ${labels.join(", ")} appeared in the model's top ${TOP_LOGPROBS} tokens.`);
  }
}
function validateGeneration(text, data) {
  try {
    const payload = text.trim().replace(/^<think>[\s\S]*?<\/think>\s*/i, "").replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/i, "$1");
    const parsed = JSON.parse(payload);
    const labels = labelsFor(data.options.length);
    const expectedKeys = data.options.map((option, index) => `${labels[index]}: ${option}`);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("expected one JSON object");
    const actualKeys = Object.keys(parsed);
    if (actualKeys.length !== expectedKeys.length || !expectedKeys.every((key) => actualKeys.includes(key))) {
      throw new Error("expected one probability for every exact option key");
    }
    const probabilities = expectedKeys.map((key) => {
      const probability = parsed[key];
      if (typeof probability !== "number" || !Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new Error("probabilities must be numbers from 0 to 1");
      }
      return probability;
    });
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    if (Math.abs(total - 1) > 0.02) throw new Error("probabilities must sum to 1");
    const index = probabilities.indexOf(Math.max(...probabilities));
    return { valid: true, choice: labels[index], choiceDescription: data.options[index], validationError: "" };
  } catch (error) {
    return { valid: false, choice: null, choiceDescription: null, validationError: error?.message ?? "invalid JSON" };
  }
}

// Warmup doubles as a sanity check with a known answer. A miscomputed GPU path (seen with MiniCPM5 and Qwen3.5 on
// wllama 3.6.1 WebGPU) still "completes", but with output like "Cimal p y" or "**Subsub** **Sub**".
const WARMUP_QUESTION = "What is the capital of France? Answer with one word.";
const WARMUP_ANSWER = /paris/i;

async function load(useLocal = false) {
  if (engine) return;
  const selected = MODEL;
  const modelUrl = useLocal ? new URL(`./assets/${selected.localFile}`, self.location.href).href : selected.url;
  const wasmUrl = new URL("./vendor/wllama/wasm/wllama.wasm", self.location.href).href;
  engine = new Wllama(
    { default: wasmUrl },
    { logger: LoggerWithoutDebug, suppressNativeLog: true, parallelDownloads: 4 },
  );
  send("loading", { message: "Fetching the model or reading it from your browser cache…" });
  const loadStart = performance.now();
  await engine.loadModelFromUrl(modelUrl, {
    n_ctx: 2048,
    n_batch: 512,
    n_gpu_layers: 999,
    cache_prompt: false,
    progressCallback: ({ loaded, total }) => send("progress", { event: {
      status: "progress", file: modelUrl, loaded, total,
      text: total ? `${(loaded / total * 100).toFixed(0)}% of model downloaded or read from cache` : "loading model",
    } }),
  });
  const loadMs = performance.now() - loadStart;
  send("loaded", { loadMs });
  send("loading", { message: "Model loaded. Compiling a real model pass…" });
  const warmupStart = performance.now();
  const warmup = await engine.createChatCompletion({
    messages: [{ role: "user", content: WARMUP_QUESTION }],
    max_tokens: 16,
    temperature: 0,
    cache_prompt: false,
    chat_template_kwargs: { enable_thinking: false },
  });
  if (!warmup?.choices?.length) throw new Error("Model warmup returned no completion.");
  const warmupText = warmup.choices[0]?.message?.content ?? "";
  if (!WARMUP_ANSWER.test(warmupText)) {
    throw new Error(`${selected.name} loaded but fails a sanity check on this GPU backend: asked for the capital of France it wrote "${warmupText.trim().slice(0, 40)}". Its output cannot be trusted here. Pick a different model and load again.`);
  }
  send("ready", { warmupMs: performance.now() - warmupStart, modelName: selected.name });
}

async function directScore(data) {
  const started = performance.now();
  const labels = labelsFor(data.options.length);
  const grammar = `root ::= ${labels.map((label) => `"${label}"`).join(" | ")}`;
  const response = await engine.createChatCompletion({
    messages: messagesFor(data, "direct"),
    max_tokens: 1,
    temperature: 1,
    top_k: 0,
    top_p: 1,
    logprobs: true,
    top_logprobs: TOP_LOGPROBS,
    grammar,
    cache_prompt: false,
    chat_template_kwargs: { enable_thinking: false },
  });
  const logits = optionLogprobs(response, labels);
  validateOptionLogprobs(logits, labels);
  const probabilities = softmax(logits);
  return {
    totalMs: performance.now() - started,
    inputTokens: response.usage?.prompt_tokens ?? 0,
    readouts: 1,
    belowCutoff: labels.filter((_, index) => logits[index] === -Infinity),
    options: data.options.map((description, index) => ({
      label: labels[index], description, probability: probabilities[index], logit: logits[index], belowCutoff: logits[index] === -Infinity,
    })),
  };
}

async function generateAnswer(data) {
  const started = performance.now();
  let firstTokenAt = null;
  let generatedText = "";
  let usage = null;
  send("generation-start");
  const stream = await engine.createChatCompletion({
    messages: messagesFor(data, "generation"),
    stream: true,
    max_tokens: 512,
    temperature: 0,
    cache_prompt: false,
    chat_template_kwargs: { enable_thinking: false },
  });
  for await (const chunk of stream) {
    const text = chunk.choices?.[0]?.delta?.content ?? "";
    if (text && firstTokenAt == null) firstTokenAt = performance.now();
    generatedText += text;
    if (chunk.usage) usage = chunk.usage;
    send("generation-update", {
      text: generatedText,
      tokens: usage?.completion_tokens ?? 0,
      ttftMs: firstTokenAt == null ? null : firstTokenAt - started,
    });
  }
  generatedText = generatedText.trim();
  return {
    generationMs: performance.now() - started,
    inputTokens: usage?.prompt_tokens ?? 0,
    ttftMs: firstTokenAt == null ? null : firstTokenAt - started,
    generatedTokens: usage?.completion_tokens ?? 0,
    generatedText,
    ...validateGeneration(generatedText, data),
  };
}

async function compare(data) {
  if (!engine) throw new Error("Load the model before running a comparison.");
  if (!Array.isArray(data.options) || data.options.length < MIN_OPTIONS || data.options.length > MAX_OPTIONS) {
    throw new Error(`This demo requires ${MIN_OPTIONS} to ${MAX_OPTIONS} options.`);
  }
  const direct = await directScore(data);
  send("direct", direct);
  const generation = await generateAnswer(data);
  send("complete", { ...generation, directMs: direct.totalMs });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await load(data.useLocal);
    if (data.type === "compare") await compare(data.data);
  } catch (error) {
    if (data.type === "load" && engine) {
      try { await engine.exit(); } catch (_) { /* best-effort cleanup after a failed load */ }
      engine = undefined;
    }
    console.error(error);
    send("error", { message: error?.message ?? String(error) });
  }
});

// Module workers with a top-level await drop messages posted before evaluation finishes; tell the page we are listening.
send("worker-ready");
