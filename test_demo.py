import re
from pathlib import Path

WEBGPU = Path(__file__).resolve().parent

def sources():
    return {name: (WEBGPU / name).read_text() for name in ("index.html", "app.js", "worker.js", "README.md", "_headers")}

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
    assert "Material+Symbols+Rounded" in text["index.html"]
    assert "Cross-Origin-Opener-Policy: same-origin" in text["_headers"]
    assert "Cross-Origin-Embedder-Policy: require-corp" in text["_headers"]

def test_single_pinned_model_autoloads():
    text = sources()
    assert "23749fefcc72300e3a2ad315e1317431b06b590a" in text["worker.js"]
    assert "23749fefcc72300e3a2ad315e1317431b06b590a" in text["README.md"]
    assert 'const MODEL = {' in text["worker.js"] and "Qwen3-0.6B-Q8_0.gguf" in text["worker.js"]
    assert "model-select" not in text["index.html"] and "modelSelect" not in text["app.js"]
    assert 'id="load" type="button" hidden' in text["index.html"]
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
    assert ">SemIf (formerly OpenJev)<" in text["index.html"]
    assert 'href="https://github.com/TheoLeeCJ/SemIf/tree/master/webgpu-demo"' in text["index.html"]
    assert "ui-mode" not in text["index.html"] and "plain-ui" not in text["app.js"]
