# SemIf browser lab

Two browser-only pages for testing small decision models locally. Nothing runs on a server: the weights come from Hugging Face into your browser cache, and your inputs never leave the page.

| Page | Live | What it shows |
|---|---|---|
| **Logits vs generation** ([`index.html`](index.html)) | https://workszop.github.io/openjev/ | Qwen3 0.6B on WebGPU answers one decision two ways: by reading its option-letter logits, and by writing a JSON distribution token by token. |
| **Vega decisions** ([`vega.html`](vega.html)) | https://workszop.github.io/openjev/vega.html | Vega 0.8B, a model that only decides, answers typed `boolean` / `choice` / `score` questions with calibrated probabilities, optionally next to Qwen3 0.6B on the same questions. |

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

## Page 2: Vega decisions (`vega.html`)

[Vega 0.8B](https://huggingface.co/nandakishorm/vega-08b-public-intents) (Nandakishor M and Convai Innovations, Apache 2.0) never generates text. A frozen Qwen3.5-0.8B reads the state and a typed question once and stops at layer 19; a 14M-parameter physics engine settles a particle into one well per option. Every answer comes with calibrated probabilities, a 90% conformal prediction set, an abstain flag and an out-of-distribution ("unbound") flag.

| Type | Criteria | Answer |
|---|---|---|
| `boolean` | `null` or `{"false": "…", "true": "…"}` | p(yes) |
| `choice` | a list of labels, or `{label: description}` | one label and the full distribution |
| `score` | ordered levels, lowest first | the most likely level and the expected level |

### Using the page

1. **load Vega · 840 MB** downloads the weights once; later visits read them from the cache and load automatically. A self-check then has to pass before anything is shown.
2. Optional: **load Qwen3 0.6B** adds page 1's model as a comparison (WebGPU, 639 MB, the same cached file as page 1). The page remembers this and reloads Qwen3 on the next visit.
3. Fill the **State** box (plain text, or a JSON object/array that is passed as structured data) and the **Questions** box (JSON). Six built-in examples, the **+ boolean / choice / score** templates and **format** help with the JSON.
4. Press **decide** (or Ctrl+Enter). Each card shows the question's answer from both models, two bars per option (yellow Vega, blue Qwen3), an agree/disagree chip and Vega's flags. The dark band counts how often the top answers agree; **Raw response JSON** has all numbers.

Questions look like this:

```json
{
  "phishing": { "type": "boolean", "instructions": "Is this email a phishing attempt?", "criteria": null },
  "route": { "type": "choice", "instructions": "Which team should handle this message?",
             "criteria": { "security": "fraud, phishing, account compromise", "billing": "invoices and payments", "support": "everything else" } },
  "severity": { "type": "score", "instructions": "How severe is this?", "criteria": ["harmless", "suspicious", "dangerous", "critical"] }
}
```

Limits: at most 8 questions per run, 2-32 options per question (Qwen3 skips questions with more than 20, since it answers with letters A-T), and 4,096 prompt tokens. The model accepts 73k, but every question re-reads the state on the CPU, so longer states are refused, never truncated. Image states, the 2- and 3-view averaging and the 4B checkpoint are not part of the page.

### Samples (`samples.txt`)

[`samples.txt`](samples.txt) has six ready-made decisions, each mixing all three question types:

1. Hotel review
2. IT helpdesk ticket
3. Polish customer complaint (reklamacja), with English questions
4. Contract clause
5. Social media comment (moderation)
6. Meeting notes as a JSON state

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

In one headless run (Chrome 152, both models loaded) the top answers agreed on 6 of the 18 sample questions. Qwen3 often put 0.85-1.0 on a single option, right or wrong, while Vega abstained on 11 of 18. Both models got some clear cases wrong, for example the Polish customer's threat to go to the consumer ombudsman. Treat the samples as probes, not a benchmark.

### How it runs

- Weights: the 8-bit ONNX conversion [`nvkudva/vega-onnx-q8`](https://huggingface.co/nvkudva/vega-onnx-q8) (`v2/`), revision `017c096d369ac698087f8c35a6ad8c867a7fe0f4`, about 840 MB for text, kept in the Cache API as `vega-weights-017c096d` (**Delete cached weights** removes it).
- Runtime: [`vega-worker.js`](vega-worker.js) runs onnxruntime-web `1.30.0` (wasm only: its WebGPU kernel cannot run 8-bit `MatMulNBits`) and `@huggingface/tokenizers` `0.2.0`, both from jsDelivr. The prompt layout, span pooling, memory chunks, thermal calibration, adapter routing and conformal sets follow the model card and the base model's `vega_api` at `views=1`. It also answers a weight-free `plan` message, so questions are validated even while the weights load.
- Self-check: after loading, the worker answers the reference case `multi_q_mixed` (all three types, a JSON state) and refuses to mark the page ready unless the token counts match, the top option matches and every probability is within 0.06 of the reference.
- Comparison: Qwen3 runs in page 1's [`worker.js`](worker.js) through its single-question `direct` message. Each decide runs Vega on every question first, then Qwen3 one question at a time (concurrent wllama completions abort). Qwen3 gets the same state, the instructions as its question, and Vega's exact option texts (`no - …` / `yes - …` for booleans, `level i - …` for scores) as options A, B, C…; its probabilities are the softmax over those letter logits, which are not calibrated.
- DOM contract: `#vega[data-vega-state]` (`idle`, `loading`, `checking`, `ready`, `error`), `data-qwen-state` (`off`, `loading`, `ready`, `error`, `unsupported`), `data-running`, `data-selfcheck` (`pending`, `pass`, `fail`), `data-run-count` and `data-agree-count`; each `.answer` card carries `data-qid`, `data-type`, `data-answer`, `data-confidence`, `data-abstain`, `data-unbound`, `data-adapter`, `data-qwen-answer`, `data-qwen-confidence` and `data-agree`; `#sample-list` carries `data-count` and `data-source`.

### Verification

Checked in headless Chrome 152 against the 27 text cases of the reference runtime's parity set ([nvkudva/vega-web](https://github.com/nvkudva/vega-web)): token ids identical on all 29 questions that fit the 4,096-token limit, same argmax 29/29, same prediction sets 29/29, abstain 28/29 (the empty-state case the model card also flags), max |Δp| 0.0503. With cross-origin isolation (8 threads) a short question takes about 1.1 s; without it, as on GitHub Pages, about 5.3 s. The same answers come out either way.

## Run locally

Web Workers and model downloads need an HTTP origin:

```bash
python3 serve.py          # prints http://127.0.0.1:<port>/ and sends the COOP/COEP headers
# or, single-threaded:
python3 -m http.server 8080
```

Open the printed URL for page 1 and add `vega.html` for page 2. Page 1 needs a WebGPU-capable browser; Vega runs on any current browser, and the Qwen3 comparison on page 2 needs WebGPU. `serve.py` makes the page cross-origin isolated, so Vega uses up to 8 CPU threads and wllama loads multi-threaded.

For deployment, any static HTTPS host works. GitHub Pages cannot send the COOP/COEP headers from [`_headers`](_headers), so there both runtimes are single-threaded: wllama loads in about 20 s from cache instead of 3 s, and Vega takes about 5 s per question instead of 1 s. Results are identical. Hosts that honour `_headers` (Cloudflare Pages, Netlify) get the faster path. `_headers` also applies `Referrer-Policy: no-referrer`, matching the page and worker policy, so cross-origin Hugging Face requests do not carry the hosting URL as a referrer.

No build step, API, database, telemetry, or server-side inference is used.

## Files

| File | Role |
|---|---|
| `index.html`, `app.js`, `worker.js` | Page 1 and its wllama worker (also serves page 2's Qwen3 comparison) |
| `vega.html`, `vega.js`, `vega-worker.js`, `vega.css` | Page 2 and its onnxruntime-web worker |
| `samples.txt` | Six sample decisions and the format for your own files |
| `tokens.css`, `style.css` | Shared edulab tokens and styles |
| `favicon.svg`, `favicon-16.png`, `favicon-32.png`, `apple-touch-icon.png`, `icon-512.png` | The "if" favicon set |
| `vendor/wllama/` | wllama 3.6.1, vendored |
| `serve.py`, `_headers` | Local server and host headers for cross-origin isolation |
| `test_demo.py` | Static checks: `python3 -m pytest -q test_demo.py` |

## Pins

- wllama `3.6.1` (vendored), Vue `3.5.21`, Lucide `0.547.0`
- onnxruntime-web `1.30.0`, `@huggingface/tokenizers` `0.2.0`
- Qwen3: [`Qwen3-0.6B Q8_0`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), revision `23749fefcc72300e3a2ad315e1317431b06b590a`, 639,446,688 bytes
- Vega: [`nvkudva/vega-onnx-q8`](https://huggingface.co/nvkudva/vega-onnx-q8), revision `017c096d369ac698087f8c35a6ad8c867a7fe0f4`

Every model URL includes an immutable Hugging Face revision. Model files remain external and are downloaded directly into browser-managed storage.

## Primary sources

- [wllama source and documentation](https://github.com/ngxson/wllama) and [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Vega base model](https://huggingface.co/nandakishorm/vega-08b-public-intents), [Vega ONNX model card](https://huggingface.co/nvkudva/vega-onnx-q8) and [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)
- Upstream fork: [TheoLeeCJ/SemIf webgpu-demo](https://github.com/TheoLeeCJ/SemIf/tree/master/webgpu-demo)

The upstream models and runtimes retain their respective licenses. This repository's original code is MIT licensed.
