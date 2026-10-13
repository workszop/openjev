import re
from pathlib import Path

WEBGPU = Path(__file__).resolve().parent

def sources():
    return {name: (WEBGPU / name).read_text() for name in ("index.html", "app.js", "worker.js", "README.md", "_headers", "vega.html", "vega.js", "vega-worker.js", "vega.css")}

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
    for name in ("vega.html", "vega.js", "vega-worker.js", "vega.css"):
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
    assert len(blocks) == 6
    for block in blocks:
        questions = json.loads(block.split("\nQUESTIONS:\n")[1])
        assert {q["type"] for q in questions.values()} == {"boolean", "choice", "score"}
    assert "\u2014" not in samples
