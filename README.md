# SemIf browser lab

This is a browser-only comparison of two readout paths through the same selected quantized local model:

1. **Direct readout** obtains wllama's log-probabilities for the allowed single-token labels and normalizes them over the displayed options. Two to twenty options need one constrained readout.
2. **Generation** greedily decodes a JSON distribution, with a 512-token limit. The model writes each full option string as a key and its estimated probability as the value.

It is a live experiment, not a prerecorded benchmark. The page displays only timings collected in the current browser session. Model loading and shader warmup are reported separately from both decision paths. The paths run sequentially to avoid WebGPU contention.

The account-support and email-triage buttons only prefill the editable inputs. They do not constrain the prompt or run the model. Options remain editable, with add/remove controls for two through twenty choices.

## Run locally

Web Workers and model downloads require an HTTP origin:

```bash
cd webgpu-demo
python3 -m http.server 8080
```

Open `http://localhost:8080` in a current WebGPU-capable browser. Expect a 639 MB download on first load. Browser caching controls repeat downloads. The page reports a clear compatibility message before any download begins.

For deployment, any static HTTPS host is sufficient. The live copy is on GitHub Pages at https://workszop.github.io/openjev/. Pages cannot send the COOP/COEP headers from `_headers`, so the page is not cross-origin isolated there and wllama loads the model single-threaded (about 20 s from cache instead of 3 s on the test machine); inference runs on WebGPU either way and results are identical. Hosts that honour `_headers` (Cloudflare Pages, Netlify) get the faster load. Locally, `python3 serve.py` sends the headers. No build step, API, database, telemetry, or server-side inference is used. wllama 3.6.1 is vendored; Vue and Lucide load from pinned CDN URLs.

Keep `_headers` when deploying to Cloudflare. It applies `Referrer-Policy: no-referrer`, matching the page and worker policy, so direct cross-origin Hugging Face asset requests do not carry the hosting URL as a referrer.

## Pins

- wllama: `3.6.1`
- Vue: `3.5.21`
- Model: [`Qwen3-0.6B Q8_0`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), revision `23749fefcc72300e3a2ad315e1317431b06b590a`, 639,446,688 bytes. It loads automatically once the WebGPU check passes.

Every model URL includes an immutable Hugging Face revision. Model files remain external and are downloaded directly into browser-managed storage.

The setup panel shows two owned balanced-accuracy scores and equal-case agreement on the selected 102-row TypeSafe subset. Those values come from the native BF16 checkpoints, not the quantized browser artifacts. The Jev comparison is TypeSafe's published value on the same subset; no live Jev endpoint was used.

## Browser verification

Chrome 152 on an RTX 3090 loaded every tier through WebGPU and completed both paths on the email-triage preset. With weights served from a local SSD to remove network variance, direct readout took 0.704 s for Qwen3-0.6B. These are operational smoke measurements from one machine, not portable performance claims.

Qwen3-0.6B is the only model and loads automatically.

MiniCPM5-2B was removed from this fork: on wllama 3.6.1 its WebGPU path produced gibberish (garbage logits, "Cmattnaaformumber…" for a trivial prompt) while the CPU path (`n_gpu_layers: 0`) was correct. Qwen3.5-4B showed the same failure on the test machine (Pop!_OS, Chrome, wllama 3.6.1): "**Subsub** **Sub**…" on the GPU, "Paris" on the CPU. It was removed as well. The warmup step now asks the model for the capital of France and refuses to mark it ready unless the answer contains "Paris". A model that fails this check is miscomputed on that GPU backend and its logits cannot be trusted.

## Measurement boundary

- **Model load** starts before wllama engine construction and ends when its model load resolves. It includes network/cache reads and GPU setup exposed by the library.
- **Warmup** measures an unreported one-token completion that compiles a real Qwen pass before the comparison.
- **Direct total** includes prompt rendering, tokenization, one grammar-constrained one-token readout, and softmax over the displayed labels.
- **Generation TTFT** starts before prompt rendering/tokenization and stops in the first token callback.
- **Generation total** uses the same start and stops after the returned answer is decoded and checked against the required JSON shape.
- **Generated tokens** come from wllama's completion usage record.

Qwen3 may prepend a `<think>...</think>` block even when thinking is disabled. The page streams that model output unchanged, removes one leading reasoning block for format validation, before parsing it internally for diagnostics. The page shows the raw model output without a validation/error banner.

The two prompts contain identical state, question, and option text. Their final format instructions differ: direct readout requests one option letter; generation asks the model to report a distribution by writing every option and probability as JSON. These generated, self-reported probabilities are a separate readout and need not match the direct token probabilities.

Direct probabilities are conditional on only the displayed label tokens. They are not calibrated probabilities, and a high value does not establish that the underlying decision is correct.

## Page 2: Vega decisions (`vega.html`)

[`vega.html`](vega.html) tests [Vega 0.8B](https://huggingface.co/nandakishorm/vega-08b-public-intents), a decision-only model: a frozen Qwen3.5-0.8B read up to layer 19 plus a 14M-parameter physics engine that answers typed `boolean`, `choice` and `score` questions with calibrated probabilities, a 90% conformal prediction set, an abstain flag and an out-of-distribution ("unbound") flag. It never generates text.

- Weights: the 8-bit ONNX conversion [`nvkudva/vega-onnx-q8`](https://huggingface.co/nvkudva/vega-onnx-q8) (`v2/`), revision `017c096d369ac698087f8c35a6ad8c867a7fe0f4`, about 840 MB for text. Downloaded on the first click of **load Vega**, kept in the Cache API (`vega-weights-017c096d`), and loaded automatically on later visits. Image states are not wired in.
- Runtime: [`vega-worker.js`](vega-worker.js) runs onnxruntime-web `1.30.0` (wasm EP only; its WebGPU kernel cannot run 8-bit `MatMulNBits`) and `@huggingface/tokenizers` `0.2.0`, both from jsDelivr. Prompt layout, span pooling, memory chunks, thermal calibration, adapter routing and conformal sets follow the model card and the base model's `vega_api` at `views=1`.
- Self-check: after loading, the worker answers the reference case `multi_q_mixed` (all three types, a JSON state) and refuses to mark the page ready unless the token counts match, the top option matches and every probability is within 0.06 of the reference.
- Comparison: **load Qwen3 0.6B** starts page 1's `worker.js` (wllama, WebGPU, same cached GGUF) in this page. Each decide then runs Vega on every question first, then Qwen3's direct readout one question at a time (concurrent wllama completions abort). Qwen3 gets the same state, the instructions as its question, and Vega's exact option texts (`no - …`/`yes - …` for booleans, `level i - …` for scores) as options A, B, C…; its probabilities are the softmax over those letter logits. Cards show both distributions and whether the top answers agree; questions with more than 20 options skip Qwen3. The choice to compare is remembered in `localStorage` and Qwen3 reloads on the next visit.
- Samples: [`samples.txt`](samples.txt) has six ready-to-paste decisions (hotel review, IT ticket, a Polish complaint, a contract clause, a moderation case and a JSON-state meeting note), each with a STATE block and a QUESTIONS JSON block.
- Limits: at most 8 questions per run, 2–32 options per question, and 4,096 prompt tokens (the model accepts 73k, but every question re-reads the state on the CPU). Longer states are refused, never truncated.
- DOM contract: `#vega[data-vega-state]` (`idle`, `loading`, `checking`, `ready`, `error`), `data-qwen-state` (`off`, `loading`, `ready`, `error`, `unsupported`), `data-running`, `data-selfcheck` (`pending`, `pass`, `fail`), `data-run-count` and `data-agree-count`; each `.answer` card carries `data-qid`, `data-type`, `data-answer`, `data-confidence`, `data-abstain`, `data-unbound`, `data-adapter`, `data-qwen-answer`, `data-qwen-confidence` and `data-agree`.

Verified in headless Chrome 152 against the 27 text cases of the reference runtime's parity set: token ids identical on all 29 questions that fit the 4,096-token limit, same argmax 29/29, same prediction sets 29/29, abstain 28/29 (the empty-state case the model card also flags), max |Δp| 0.0503. With cross-origin isolation (8 threads) a short question takes about 1.1 s; without it, as on GitHub Pages, about 5.3 s.

## Primary sources

- [wllama source and documentation](https://github.com/ngxson/wllama)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Vega ONNX model card](https://huggingface.co/nvkudva/vega-onnx-q8) and [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)

The upstream model and runtime retain their respective licenses. This repository's original code is MIT licensed.
