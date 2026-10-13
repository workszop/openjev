import re
from pathlib import Path

WEBGPU = Path(__file__).resolve().parent

def sources():
    return {name: (WEBGPU / name).read_text() for name in ("index.html", "app.js", "worker.js", "README.md", "_headers", "vega.html", "vega.js", "vega-worker.js", "vega.css", "basal-worker.js", "py-json.js")}

def test_static_runtime_and_pins():
    text = sources()
    assert 'src="app.js"' in text["index.html"]
    assert 'href="style.css?v=' in text["index.html"]
    assert 'new Worker("worker.js", { type: "module" })' in text["app.js"]
    assert 'import("./vendor/wllama/index.js")' in text["worker.js"]
    assert (WEBGPU / "vendor/wllama/wasm/wllama.wasm").stat().st_size > 1_000_000
    assert (WEBGPU / "vendor/wllama/LICENCE").is_file()
    assert "wllama" in text["README.md"] and "3.6.1" in text["README.md"]
    assert "vue@3.5.21" in text["app.js"]
    assert "lucide@0.547.0/dist/umd/lucide.min.js" in text["index.html"] and "Material" not in text["index.html"]
    assert "Cross-Origin-Opener-Policy: same-origin" in text["_headers"]
    assert "Cross-Origin-Embedder-Policy: require-corp" in text["_headers"]

def test_single_pinned_model_autoloads():
    text = sources()
    assert "23749fefcc72300e3a2ad315e1317431b06b590a" in text["worker.js"]
    assert "23749fefcc72300e3a2ad315e1317431b06b590a" in text["README.md"]
    assert 'const MODEL = {' in text["worker.js"] and "Qwen3-0.6B-Q8_0.gguf" in text["worker.js"]
    assert "model-select" not in text["index.html"] and "modelSelect" not in text["app.js"]
    assert 'id="load" type="button" class="btn" hidden' in text["index.html"]
    assert "startLoad();" in text["app.js"]  # auto-load right after the WebGPU check
    assert 'loadButton.addEventListener("click", startLoad)' in text["app.js"]
    assert "WARMUP_QUESTION" in text["worker.js"] and "/paris/i" in text["worker.js"]
    assert 'id="quality-title"' in text["index.html"]
    assert "Published Jev" in text["index.html"]
    for score in ("44.0%", "52.8%", "40.7%", "88.3%"):
        assert score in text["index.html"]
    assert "Qwen3.5" not in text["index.html"] and "MiniCPM" not in text["index.html"]

def test_live_comparison_and_limits():
    text = sources()
    combined = "\n".join(text.values())
    assert "performance.now()" in combined
    assert "createChatCompletion" in text["worker.js"]
    assert "logprobs: true" in text["worker.js"]
    assert "top_logprobs: TOP_LOGPROBS" in text["worker.js"]
    assert "grammar" in text["worker.js"]
    assert "stream: true" in text["worker.js"]
    assert "JSON.parse" in text["worker.js"]
    assert "<think>" in text["worker.js"]
    assert "probabilities must sum to 1" in text["worker.js"]
    assert "Route north" in text["worker.js"]
    assert "const MAX_OPTIONS = 20" in text["app.js"]
    assert "const MIN_OPTIONS = 2" in text["app.js"]
    for phrase in ("conditional probabilities", "not calibrated", "sequential", "warmup"):
        assert phrase in combined.lower()
    assert "WebSocket" not in combined
    assert not re.search(r"/api/(?:generate|score)", combined)

def test_identity_notice_and_ui_mode_switch():
    text = sources()
    assert 'class="edu-logo"' in text["index.html"] and "SemIf browser lab" in text["index.html"]
    assert 'href="https://github.com/TheoLeeCJ/SemIf/tree/master/webgpu-demo"' in text["index.html"]
    assert "ui-mode" not in text["index.html"] and "plain-ui" not in text["app.js"]

def test_vega_page_pins_and_contract():
    text = sources()
    worker = text["vega-worker.js"]
    assert 'href="vega.html"' in text["index.html"] and 'href="index.html"' in text["vega.html"]
    assert 'src="vega.js"' in text["vega.html"] and 'href="vega.css?v=' in text["vega.html"]
    assert 'new Worker("vega-worker.js", { type: "module" })' in text["vega.js"]
    assert "onnxruntime-web@${ORT_VERSION}" in worker and 'ORT_VERSION = "1.30.0"' in worker
    assert "@huggingface/tokenizers@0.2.0" in worker
    assert "017c096d369ac698087f8c35a6ad8c867a7fe0f4" in worker and "017c096d369ac698087f8c35a6ad8c867a7fe0f4" in text["README.md"]
    assert 'executionProviders: ["wasm"]' in worker and "vision_q8" not in worker
    assert 'send("worker-ready")' in worker and "await workerReady" in text["vega.js"]
    assert "selfCheck" in worker and "tolerance: 0.06" in worker
    assert "const BROWSER_MAX_TOKENS = 4096" in worker
    for attr in ("data-vega-state", "data-selfcheck", "data-run-count"):
        assert attr in text["vega.html"]
    for key in ("qid", "type", "answer", "confidence", "abstain", "unbound", "adapter"):
        assert key in text["vega.js"]
    assert 'aria-live="polite"' in text["vega.html"] and "Control+Enter" in text["vega.html"]
    for name in ("vega.html", "vega.js", "vega-worker.js", "vega.css", "basal-worker.js", "py-json.js"):
        assert "\u2014" not in text[name]

def test_vega_qwen_comparison_and_samples():
    import json
    text = sources()
    assert 'new Worker("worker.js", { type: "module" })' in text["vega.js"]
    assert 'data.type === "direct"' in text["worker.js"] and 'send("direct-result"' in text["worker.js"]
    assert 'id="load-qwen"' in text["vega.html"] and 'data-qwen-state="off"' in text["vega.html"]
    assert "const QWEN_MAX_OPTIONS = 20" in text["vega.js"] and "function nextQwen()" in text["vega.js"]
    assert 'if (data.type === "plan")' in text["vega-worker.js"]
    assert 'href="samples.txt"' in text["vega.html"]
    samples = (WEBGPU / "samples.txt").read_text()
    blocks = samples.split("\n--- ")[1:]
    assert len(blocks) == 12
    for block in blocks:
        questions = json.loads(block.split("\nQUESTIONS:\n")[1])
        assert {q["type"] for q in questions.values()} == {"boolean", "choice", "score"}
    assert "\u2014" not in samples

def test_samples_loader_button():
    text = sources()
    assert 'id="load-samples"' in text["vega.html"] and 'id="sample-list"' in text["vega.html"]
    assert 'const SAMPLES_URL = "samples.txt"' in text["vega.js"]
    assert "function parseSamples(text, fileName)" in text["vega.js"] and 'samplesButton.addEventListener("click", loadSamples)' in text["vega.js"]
    assert 'id="sample-file" type="file" accept=".txt,text/plain"' in text["vega.html"] and 'id="open-samples"' in text["vega.html"]
    assert "const MAX_SAMPLE_FILE_BYTES = 2_000_000" in text["vega.js"] and "file.text()" in text["vega.js"]

def test_favicon_linked_on_both_pages():
    text = sources()
    for page in ("index.html", "vega.html"):
        assert 'rel="icon" type="image/svg+xml" href="favicon.svg?v=' in text[page]
        assert 'rel="apple-touch-icon" sizes="180x180"' in text[page]
    svg = (WEBGPU / "favicon.svg").read_text()
    assert 'viewBox="0 0 64 64"' in svg and ">if</text>" in svg and "--" not in svg.split("<style>")[0]
    for name in ("favicon-16.png", "favicon-32.png", "apple-touch-icon.png", "icon-512.png"):
        assert (WEBGPU / name).stat().st_size > 200

def test_basal_third_model_and_picker():
    text = sources()
    worker = text["basal-worker.js"]
    assert 'new Worker("basal-worker.js", { type: "module" })' in text["vega.js"]
    assert 'import { pyJsonDumps } from "./py-json.js";' in worker and 'import { pyJsonDumps } from "./py-json.js";' in text["vega-worker.js"]
    assert "export function pyJsonDumps" in text["py-json.js"]
    assert 'ORT_VERSION = "1.30.0"' in worker and "ort.webgpu.min.mjs" in worker and 'executionProviders: ["webgpu"]' in worker
    # fp32 activations: the fp16 WebGPU graph miscomputed on Intel Xe-LPG
    assert 'new ort.Tensor("float32", new Float32Array(0), [1, KV_HEADS, 0, HEAD_DIM])' in worker
    assert """const PREFILL = '{"answer": "';""" in worker and 'const LETTERS = "ABCDEFGHIJ";' in worker
    revision = re.search(r'const MODEL_REVISION = "([0-9a-f]{40})";', worker)
    assert revision and revision.group(1) in text["README.md"]
    assert 'MODEL_REPO = "andrzej/basal-1.5-mini-onnx-web"' in worker
    assert "selfCheck" in worker and "tolerance: 0.05" in worker and 'send("worker-ready")' in worker
    assert "const BROWSER_MAX_TOKENS = 4096" in worker and "PREFILL_CHUNK" in worker
    for model in ("vega", "basal", "qwen"):
        assert f'data-pick="{model}"' in text["vega.html"] and f'data-model="{model}"' in text["vega.html"]
    assert 'id="load-basal"' in text["vega.html"] and 'data-basal-state="off"' in text["vega.html"] and 'data-basal-selfcheck="pending"' in text["vega.html"]
    assert 'const MODEL_ORDER = ["vega", "basal", "qwen"];' in text["vega.js"] and "function nextModel()" in text["vega.js"]
    assert 'const PICK_KEY = "vegaLab.models";' in text["vega.js"] and "runnableModels()" in text["vega.js"]
    assert ".swatch-basal, .bar-basal i" in text["vega.css"]

def test_vega_page_never_autoloads():
    text = sources()
    js = text["vega.js"]
    init = js.split("// ─── Init ───")[1]
    assert "startLoad" not in init and "startQwenLoad" not in init and "startBasalLoad" not in init and "loadSelected" not in init
    for probe in js.split('case "probe":')[1:]:
        assert "startLoad()" not in probe.split("break;")[0] and "startBasalLoad()" not in probe.split("break;")[0]
    assert 'id="load-selected"' in text["vega.html"] and 'loadSelectedButton.addEventListener("click", loadSelected)' in js

def test_polish_samples_are_fully_polish():
    import json
    polish = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    blocks = (WEBGPU / "samples.txt").read_text().split("\n--- ")[7:]
    assert len(blocks) == 6
    for block in blocks:
        state = block.split("\nSTATE:\n")[1].split("\nQUESTIONS:\n")[0]
        questions = json.loads(block.split("\nQUESTIONS:\n")[1])
        assert polish & set(state) and polish & set(json.dumps(questions, ensure_ascii=False))
        for q in questions.values():
            options = q["criteria"] if isinstance(q["criteria"], list) else list((q["criteria"] or {}).values())
            assert len(options) <= 10
