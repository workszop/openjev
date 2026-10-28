# SemIf browser lab

Two browser-only pages for testing small decision models locally. Nothing runs on a server: the weights come from Hugging Face into your browser cache, and your inputs never leave the page.

| Page | Live | What it shows |
|---|---|---|
| **Logits vs generation** ([`index.html`](index.html)) | https://workszop.github.io/openjev/ | Qwen3 0.6B on WebGPU answers one decision two ways: by reading its option-letter logits, and by writing a JSON distribution token by token. |
| **d1 decisions** ([`vega.html`](vega.html)) | https://workszop.github.io/openjev/vega.html | d1-2B from Liquid AI, a model that only decides, answers typed `boolean` / `choice` / `score` questions with calibrated probabilities, every question about one state in one forward pass. Tick basal-1.5-mini and/or Qwen3 0.6B to compare them on the same questions. |

A header nav switches between the pages. Both use the edulab look ([`tokens.css`](tokens.css), [`style.css`](style.css)) and the yellow "if" favicon.

## Page 1: Logits vs generation (`index.html`)

A comparison of two readout paths through the same quantized local model:

1. **Direct readout** obtains wllama's log-probabilities for the allowed single-token labels and normalizes them over the displayed options. Two to twenty options need one constrained readout.
2. **Generation** greedily decodes a JSON distribution, with a 512-token limit. The model writes each full option string as a key and its estimated probability as the value.

It is a live experiment, not a prerecorded benchmark. The page displays only timings collected in the current browser session. Model loading and shader warmup are reported separately from both decision paths. The paths run sequentially to avoid WebGPU contention.

The account-support and email-triage buttons only prefill the editable inputs. They do not constrain the prompt or run the model. Options remain editable, with add/remove controls for two through twenty choices.

The setup panel shows two owned balanced-accuracy scores and equal-case agreement on the selected 102-row TypeSafe subset. Those values come from the native BF16 checkpoints, not the quantized browser artifacts. The Jev comparison is TypeSafe's published value on the same subset; no live Jev endpoint was used.

### Measurement boundary

- **Model load** starts before wllama engine construction and ends when its model load resolves. It includes network/cache reads and GPU setup exposed by the library.
- **Warmup** measures an unreported one-token completion that compiles a real Qwen pass before the comparison.
- **Direct total** includes prompt rendering, tokenization, one grammar-constrained one-token readout, and softmax over the displayed labels.
- **Generation TTFT** starts before prompt rendering/tokenization and stops in the first token callback.
- **Generation total** uses the same start and stops after the returned answer is decoded and checked against the required JSON shape.
- **Generated tokens** come from wllama's completion usage record.

Qwen3 may prepend a `<think>...</think>` block even when thinking is disabled. The page streams that model output unchanged, removes one leading reasoning block for format validation, before parsing it internally for diagnostics. The page shows the raw model output without a validation/error banner.

The two prompts contain identical state, question, and option text. Their final format instructions differ: direct readout requests one option letter; generation asks the model to report a distribution by writing every option and probability as JSON. These generated, self-reported probabilities are a separate readout and need not match the direct token probabilities.

Direct probabilities are conditional on only the displayed label tokens. They are not calibrated probabilities, and a high value does not establish that the underlying decision is correct.

### Model checks

Chrome 152 on an RTX 3090 completed both paths on the email-triage preset. With weights served from a local SSD to remove network variance, direct readout took 0.704 s for Qwen3-0.6B. These are operational smoke measurements from one machine, not portable performance claims.

Qwen3-0.6B is the only model and loads automatically. MiniCPM5-2B was removed from this fork: on wllama 3.6.1 its WebGPU path produced gibberish (garbage logits, "Cmattnaaformumber…" for a trivial prompt) while the CPU path (`n_gpu_layers: 0`) was correct. Qwen3.5-4B showed the same failure on the test machine (Pop!_OS, Chrome, wllama 3.6.1): "**Subsub** **Sub**…" on the GPU, "Paris" on the CPU. It was removed as well. The warmup step asks the model for the capital of France and refuses to mark it ready unless the answer contains "Paris". A model that fails this check is miscomputed on that GPU backend and its logits cannot be trusted.

## Page 2: d1 decisions (`vega.html`)

[d1-2B](https://huggingface.co/LiquidAI/d1-2B) (Liquid AI, built on LFM3-VL-2B, [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-2B/blob/main/LICENSE)) never generates text. It reads the state and up to 16 typed questions in one forward pass and takes each answer from the logits at that question's answer slot, softmaxed over the option codes and divided by the temperature the release ships. Every answer is a calibrated distribution over the options you listed, so it can never fall outside them. (The page kept its `vega.html` address; it ran Vega 0.8B before d1-2B replaced it.)

| Type | Criteria | Answer |
|---|---|---|
| `boolean` | `null` or `{"false": "…", "true": "…"}` | p(yes) |
| `choice` | a list of labels, or `{label: description}` | one label and the full distribution |
| `score` | 2 to 10 ordered levels, lowest first | the median level (d1's own answer), the most likely level and the expected level |

### Using the page

1. **Pick the models.** Three cards, each with a tick box and a **load** button: **d1-2B** (WebGPU, 1.7 GB), **basal-1.5-mini** (WebGPU, 1.5 GB) and **Qwen3 0.6B** (WebGPU, 639 MB, the same cached file as page 1). Nothing loads by itself: **load selected** loads every ticked model, and each card's **load** loads just that one (and ticks it). Only ticked, loaded models decide. Weights download once and stay in the browser cache, so later loads skip the download; the cards say which models are cached. Each model has to pass a self-check before it is marked ready. The page remembers the ticks.
2. Fill the **State** box (plain text, or a JSON object/array that is passed as structured data) and the **Questions** box (JSON). Six built-in examples, the **+ boolean / choice / score** templates and **format** help with the JSON.
3. Press **decide** (or Ctrl+Enter). Each card has one lane per model that ran, one bar per model and option (yellow d1, green basal, blue Qwen3), a chip ("agree", "2 of 3 agree", "disagree"), d1's median level for scores and basal's 5%-risk verdict. The dark band counts how often all models agree on the top answer; **Raw response JSON** has all numbers.

Questions look like this:

```json
{
  "phishing": { "type": "boolean", "instructions": "Is this email a phishing attempt?", "criteria": null },
  "route": { "type": "choice", "instructions": "Which team should handle this message?",
             "criteria": { "security": "fraud, phishing, account compromise", "billing": "invoices and payments", "support": "everything else" } },
  "severity": { "type": "score", "instructions": "How severe is this?", "criteria": ["harmless", "suspicious", "dangerous", "critical"] }
}
```

Limits: at most 16 questions per run (d1's trained set size, so one pass), 2-32 options per choice and 2-10 levels per score (basal skips questions with more than 10 options and Qwen3 with more than 20, since they answer with letters A-J and A-T), and 4,096 tokens for the state and all questions together, the model card's WebGPU limit for `decoder_q4`. Longer inputs are refused, never truncated. Pictures, video, multiple choice (`multiple: true`) and d1's typed reads (text, number, date, box) are not part of the page.

### Samples (`samples.txt`)

[`samples.txt`](samples.txt) has twelve ready-made decisions, each mixing all three question types. Samples 7-12 are entirely in Polish: Polish state and Polish questions, option keys and descriptions (basal then reads them with its Polish prompt).

1. Hotel review
2. IT helpdesk ticket
3. Polish customer complaint (reklamacja), with English questions
4. Contract clause
5. Social media comment (moderation)
6. Meeting notes as a JSON state
7. Wezwanie z urzędu skarbowego (a 7-day deadline to count from the dates)
8. Awaria systemu kasowego w sieci sklepów (queue, priority, scope)
9. Opinia o zakupie w sklepie internetowym (product vs shop)
10. Żądanie dotyczące danych osobowych (RODO, several requests in one email)
11. SMS o dopłacie do przesyłki (parcel-fee phishing)
12. Zgłoszenie szkody z polisy, a JSON state with Polish keys (policy dates vs the date of the incident)

**load samples.txt** turns them into buttons that fill both boxes. **open my .txt…** does the same for a file of your own, read in the browser and never uploaded (up to 2 MB, Windows line endings are fine). The format:

```text
--- 1. My first sample ---
optional note

STATE:
The text to decide on (or a JSON object).

QUESTIONS:
{ "urgent": { "type": "boolean", "instructions": "Is this urgent?", "criteria": null } }
```

Each sample is a header line starting with `--- `, a line with only `STATE:` and a line with only `QUESTIONS:` followed by one JSON object. A file with a single `STATE:` / `QUESTIONS:` pair and no header also works; its button is named after the file. Samples whose questions are not valid JSON are skipped, and the page names them and says why.

Treat the samples as probes, not a benchmark.

### How it runs

- Weights: the official ONNX export [`LiquidAI/d1-2B-ONNX`](https://huggingface.co/LiquidAI/d1-2B-ONNX), revision `7f5c6801cba6164dda9daae925178f89717b960f`, graph `onnx/decoder_q4.onnx` with its 1.74 GB `decoder_q4.onnx_data` (int4 MLP gate and up projections, int8 elsewhere, **fp32 activations**), plus `tokenizer.json` and `calibration.json`, kept in the Cache API as `d1-weights-7f5c6801` (**Delete cached weights** removes it). The vision encoder is not downloaded; the decoder gets empty `image_features` and `video_features`. The fp16-activation graphs (`q4f16`, `q8f16`, `fp16`), which the model card recommends for GPUs, are not used: 16-bit WebGPU kernels miscomputed on an Intel Xe-LPG with basal.
- Runtime: [`d1-worker.js`](d1-worker.js) runs onnxruntime-web `1.30.0` with WebGPU (`ort.webgpu.min.mjs`, as the model card requires) and `@huggingface/tokenizers` `0.2.0`, both from jsDelivr. It ports the checkpoint's own `prompt.py`, `attention.py` (`pack_set`) and `runner.py` for `noul`, `choice` and `score` questions: the ChatML prompt with a `<think></think>` lead, the state as text or `json.dumps(indent=2)`, special tokens in user text broken with a zero-width space, option codes from scheme 2 (`d1_codes`) with the fallback pool, every question packed after the state with `question_ids` (-1 on the state, k on question k) and `slot_positions` at each question's last token, one forward pass, then the max over each option's token spellings divided by `calibration.json`'s default temperature (1.337) and softmaxed. A page `boolean` is d1's `noul`; a score's own answer is the median level. The weight file is read into one buffer sized from `content-length`, so 1.7 GB is never held twice. The worker also answers a weight-free `plan` message, so questions are validated even while the weights load.
- Self-check: after loading, the worker answers the model card's quick-start ticket (a refund yes/no, a three-team choice, a three-level urgency score) and refuses to mark d1 ready unless the token count and the first 24 token ids match, the top options match and every probability is within 0.05 of the reference.
- Comparison: Qwen3 runs in page 1's [`worker.js`](worker.js) through its single-question `direct` message. Each decide runs d1 on every question first, then basal, then Qwen3 one question at a time (concurrent wllama completions abort). Qwen3 gets the same state, the instructions as its question, and the plan's option texts (`no - …` / `yes - …` for booleans, `level i - …` for scores) as options A, B, C…; its probabilities are the softmax over those letter logits, which are not calibrated.
- basal: [`basal-1.5-mini`](https://github.com/rkinas/basal) (Remigiusz Kinas, Apache 2.0, fine-tuned from Bielik-1.5B) has no official browser build, so this page uses an ONNX conversion published as [`andrzej/basal-1.5-mini-onnx-web`](https://huggingface.co/andrzej/basal-1.5-mini-onnx-web), revision `a41eff3a57bd53ea99766f2bb6580b3751255284`, about 1.5 GB, cached as `basal-weights-<rev>`. [`basal-worker.js`](basal-worker.js) runs it on onnxruntime-web `1.30.0` with WebGPU. It renders basal's own prompt (Polish or English system message, options lettered A-J, the answer started as `{"answer": "`), reads the softmax over the option-letter logits, asks in both option orders, averages, and applies basal's per-type temperature, as in `basal/prompt.py` and `basal/server.py`. The system message and state are computed once per decide and each question reuses that KV cache; prompts are prefilled in 256-token chunks. A page `choice` option becomes `key: description`, a `boolean` becomes Yes / No (Tak / Nie), a `score` keeps its levels. "accept" means the confidence clears basal's certified 5%-risk threshold.
- basal conversion: asymmetric int4 with int8 for the most sensitive layers, fp32 output head and **fp32 activations**. The fp16 WebGPU graph answered almost every prompt with the same option on an Intel Xe-LPG GPU, while onnxruntime-web on wasm and the fp32 graph on the same GPU matched the reference. On basal's 24 example items in both option orders the port keeps the fp32 original's top answer on 47 of 48 prompts (mean max |Δp| 0.069 on WebGPU). The self-check repeats three of them (choice, yes/no, score; PL and EN) within 0.05 before basal is marked ready.
- DOM contract: `#vega[data-d1-state]`, `data-qwen-state` and `data-basal-state` (`off`, `loading`, `checking`, `ready`, `error`, `unsupported`), `data-selected` (the ticked models), `data-running`, `data-d1-selfcheck` and `data-basal-selfcheck` (`pending`, `pass`, `fail`), `data-run-count` and `data-agree-count`; each `.model-card` carries `data-model`, `data-state` and `data-selected`; each `.answer` card carries `data-qid`, `data-type`, `data-d1-answer`, `data-d1-confidence`, `data-d1-level` (scores), `data-basal-answer`, `data-basal-confidence`, `data-basal-accepted`, `data-qwen-answer`, `data-qwen-confidence` and `data-agree` (all models that answered agree); `#results[data-models]` lists the lanes; `#sample-list` carries `data-count` and `data-source`.

### Verification

d1 was checked in headless Chrome on an Intel Xe-LPG (WebGPU) against onnxruntime CPU running the same `decoder_q4` graph, with prompts built by the checkpoint's own `prompt.py`, over the 6 presets and the 12 samples (52 questions, 77-392 tokens a pass): token ids identical in all 18 rows, same top answer 52/52, mean |Δp| 0.013, max 0.11 (on a near-tie, 0.41 vs 0.49 on the CPU). The CPU reference reproduces the model card's quick start (refund 1.0, billing, medium). One pass over 3 questions takes about 1 s once the shaders are warm (the first pass after loading, about 6 s); loading from the cache takes about 25 s.

## Run locally

Web Workers and model downloads need an HTTP origin:

```bash
python3 serve.py          # prints http://127.0.0.1:<port>/ and sends the COOP/COEP headers
# or, single-threaded:
python3 -m http.server 8080
```

Open the printed URL for page 1 and add `vega.html` for page 2. Both pages need a WebGPU-capable browser. To test local weights, put them in the gitignored `local-models/` folder and open `vega.html?d1Base=local-models/<folder>` (the same layout as the Hugging Face repo, `onnx/` included) or `vega.html?basalBase=local-models/<folder>`. `serve.py` makes the page cross-origin isolated, so wllama loads multi-threaded.

For deployment, any static HTTPS host works. GitHub Pages cannot send the COOP/COEP headers from [`_headers`](_headers), so there wllama is single-threaded and loads in about 20 s from cache instead of 3 s. Results are identical. Hosts that honour `_headers` (Cloudflare Pages, Netlify) get the faster path. `_headers` also applies `Referrer-Policy: no-referrer`, matching the page and worker policy, so cross-origin Hugging Face requests do not carry the hosting URL as a referrer.

No build step, API, database, telemetry, or server-side inference is used.

## Files

| File | Role |
|---|---|
| `index.html`, `app.js`, `worker.js` | Page 1 and its wllama worker (also serves page 2's Qwen3 comparison) |
| `vega.html`, `vega.js`, `vega.css` | Page 2 |
| `d1-worker.js` | Page 2's d1 worker (onnxruntime-web, WebGPU) |
| `basal-worker.js` | Page 2's basal worker (onnxruntime-web, WebGPU) |
| `py-json.js` | Python-style `json.dumps`, shared by both decision workers |
| `samples.txt` | Twelve sample decisions (six in Polish) and the format for your own files |
| `tokens.css`, `style.css` | Shared edulab tokens and styles |
| `favicon.svg`, `favicon-16.png`, `favicon-32.png`, `apple-touch-icon.png`, `icon-512.png` | The "if" favicon set |
| `vendor/wllama/` | wllama 3.6.1, vendored |
| `serve.py`, `_headers` | Local server and host headers for cross-origin isolation |
| `test_demo.py` | Static checks: `python3 -m pytest -q test_demo.py` |

## Pins

- wllama `3.6.1` (vendored), Vue `3.5.21`, Lucide `0.547.0`
- onnxruntime-web `1.30.0`, `@huggingface/tokenizers` `0.2.0`
- Qwen3: [`Qwen3-0.6B Q8_0`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), revision `23749fefcc72300e3a2ad315e1317431b06b590a`, 639,446,688 bytes
- d1: [`LiquidAI/d1-2B-ONNX`](https://huggingface.co/LiquidAI/d1-2B-ONNX), revision `7f5c6801cba6164dda9daae925178f89717b960f`, `onnx/decoder_q4.onnx`
- basal: [`andrzej/basal-1.5-mini-onnx-web`](https://huggingface.co/andrzej/basal-1.5-mini-onnx-web), revision `a41eff3a57bd53ea99766f2bb6580b3751255284`, converted from [`Remek/basal-1.5-mini`](https://huggingface.co/Remek/basal-1.5-mini)

Every model URL includes an immutable Hugging Face revision. Model files remain external and are downloaded directly into browser-managed storage.

## Primary sources

- [wllama source and documentation](https://github.com/ngxson/wllama) and [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [basal engine](https://github.com/rkinas/basal) and [basal-1.5-mini](https://huggingface.co/Remek/basal-1.5-mini)
- [d1-2B model card](https://huggingface.co/LiquidAI/d1-2B), [d1-2B ONNX model card](https://huggingface.co/LiquidAI/d1-2B-ONNX) and [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)
- Upstream fork: [TheoLeeCJ/SemIf webgpu-demo](https://github.com/TheoLeeCJ/SemIf/tree/master/webgpu-demo)

The upstream models and runtimes retain their respective licenses. This repository's original code is MIT licensed.
