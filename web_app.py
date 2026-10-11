"""Local web interface for the Logion Greek philology tools."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from logion import (
    GapFiller,
    GapFillerConfig,
    GapLengthEstimator,
    ScribalErrorConfig,
    ScribalErrorDetector,
)

MODEL_NAME = "wantuta/bert_ancient_greek_mlm"
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "length_estimator.pt"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("logion-web")


class LogionService:
    """Owns the shared tokenizer, model, and Logion pipelines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.filler: GapFiller | None = None
        self.detector: ScribalErrorDetector | None = None
        self._load()

    def _load(self) -> None:
        try:
            logger.info("Loading %s ...", MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
            model.eval()

            estimator = GapLengthEstimator()
            if ARTIFACT_PATH.exists():
                estimator.load(str(ARTIFACT_PATH))

            self.filler = GapFiller(
                {size: model for size in (2, 3, 4, 5)},
                tokenizer,
                estimator,
                base_model=model,
                device="cpu",
                config=GapFillerConfig(beam_size=3, top_k=3, num_results=10),
            )
            self.detector = ScribalErrorDetector(
                model,
                tokenizer,
                device="cpu",
                config=ScribalErrorConfig(),
            )
            self.ready = True
            logger.info("Logion web service is ready")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Unable to load the Logion model")

    def fill_gap(self, context: str, n_chars: int) -> list[dict[str, Any]]:
        if not self.filler:
            raise RuntimeError(self.error or "The model is not ready.")
        if not context.strip():
            raise ValueError("Context is required.")
        if not 3 <= n_chars <= 10:
            raise ValueError("Gap length must be between 3 and 10 characters.")
        return [suggestion.as_dict() for suggestion in self.filler.fill_gap(context, n_chars)]

    def detect_errors(self, text: str, mode: str) -> list[dict[str, Any]]:
        if not self.detector:
            raise RuntimeError(self.error or "The model is not ready.")
        if not text.strip():
            raise ValueError("Text is required.")
        if mode not in {"threshold", "ratio"}:
            raise ValueError("Mode must be 'threshold' or 'ratio'.")
        return [flag.as_dict() for flag in self.detector.detect(text, mode=mode)]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Logion — Greek Philology</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #f4f1ea; color: #202124; }
    main { max-width: 1050px; margin: 0 auto; padding: 32px 18px 56px; }
    h1 { margin-bottom: 4px; color: #3d2d4f; }
    .subtitle { margin-top: 0; color: #665c6d; }
    .status { padding: 12px 16px; border-radius: 10px; background: #fff; margin: 20px 0; }
    .status.ready { border-left: 5px solid #218739; }
    .status.loading { border-left: 5px solid #d28b18; }
    .status.error { border-left: 5px solid #b3261e; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
    .card { background: #fff; border-radius: 14px; padding: 20px; box-shadow: 0 2px 10px #3d2d4f12; }
    textarea, input, select { width: 100%; box-sizing: border-box; margin: 8px 0 14px; padding: 10px; border: 1px solid #c9c1ce; border-radius: 8px; font: inherit; }
    textarea { min-height: 115px; resize: vertical; }
    button { background: #3d2d4f; color: white; border: 0; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    button:disabled { opacity: .55; cursor: wait; }
    .result { margin-top: 16px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .92rem; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }
    th { color: #665c6d; }
    .hint { color: #665c6d; font-size: .9rem; }
    .error { color: #b3261e; white-space: pre-wrap; }
    .loading-note { color: #9b650f; }
  </style>
</head>
<body>
<main>
  <h1>Logion</h1>
  <p class="subtitle">Machine learning tools for premodern Greek philology</p>
  <div id="status" class="status loading">Loading the Ancient Greek model...</div>
  <div class="grid">
    <section class="card">
      <h2>Fill a gap</h2>
      <p class="hint">Use <code>[MASK]</code> to mark the missing span.</p>
      <label for="gap-context">Context</label>
      <textarea id="gap-context">Οὐκ οἶδα [MASK] πότερον</textarea>
      <label for="gap-length">Missing characters (3–10)</label>
      <input id="gap-length" type="number" min="3" max="10" value="4">
      <button id="fill-button" onclick="fillGap()" disabled>Suggest reconstructions</button>
      <div id="gap-result" class="result"></div>
    </section>
    <section class="card">
      <h2>Detect scribal errors</h2>
      <label for="error-text">Greek text</label>
      <textarea id="error-text">Οὗτος δὶς βασιλεύσας ἠύχετο καὶ τρὶς.</textarea>
      <label for="error-mode">Ranking mode</label>
      <select id="error-mode">
        <option value="threshold">Threshold flags</option>
        <option value="ratio">Chance-confidence ratio</option>
      </select>
      <button id="error-button" onclick="detectErrors()" disabled>Analyze text</button>
      <div id="error-result" class="result"></div>
    </section>
  </div>
</main>
<script>
const statusBox = document.getElementById("status");
async function updateStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (status.ready) {
      statusBox.className = "status ready";
      statusBox.textContent = "Model ready.";
      document.querySelectorAll("button").forEach(button => button.disabled = false);
    } else if (status.error) {
      statusBox.className = "status error";
      statusBox.textContent = "Model error: " + status.error;
    } else {
      statusBox.className = "status loading";
      statusBox.textContent = "Loading the Ancient Greek model...";
      setTimeout(updateStatus, 1500);
    }
  } catch (error) {
    statusBox.className = "status error";
    statusBox.textContent = "Could not reach the local service: " + error;
  }
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[char]));
}
function showError(target, error) {
  target.innerHTML = `<p class="error">${escapeHtml(error)}</p>`;
}
async function fillGap() {
  const target = document.getElementById("gap-result");
  target.innerHTML = "<p>Searching...</p>";
  try {
    const response = await fetch("/api/fill", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        context: document.getElementById("gap-context").value,
        n_chars: Number(document.getElementById("gap-length").value)
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    target.innerHTML = `<table><tr><th>Suggestion</th><th>Tokens</th><th>Score</th></tr>${
      payload.results.map(item => `<tr><td>${escapeHtml(item.text)}</td><td>${item.num_tokens}</td><td>${Number(item.score).toExponential(3)}</td></tr>`).join("")
    }</table>`;
  } catch (error) { showError(target, error.message); }
}
async function detectErrors() {
  const target = document.getElementById("error-result");
  target.innerHTML = "<p>Analyzing...</p>";
  try {
    const response = await fetch("/api/detect", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text: document.getElementById("error-text").value,
        mode: document.getElementById("error-mode").value
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    target.innerHTML = payload.results.length ? `<table><tr><th>Token</th><th>Suggestion</th><th>Chance</th><th>Confidence</th><th>Distance</th></tr>${
      payload.results.map(item => `<tr><td>${escapeHtml(item.token)}</td><td>${escapeHtml(item.suggestion)}</td><td>${Number(item.chance).toExponential(3)}</td><td>${Number(item.confidence).toFixed(3)}</td><td>${Number(item.scribal_distance).toFixed(2)}</td></tr>`).join("")
    }</table>` : "<p>No likely scribal errors found.</p>";
  } catch (error) { showError(target, error.message); }
}
updateStatus();
</script>
</body>
</html>"""


class RequestHandler(BaseHTTPRequestHandler):
    service: LogionService

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._send_json({"ready": self.service.ready, "error": self.service.error})
        else:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/fill", "/api/detect"}:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            with self.service._lock:
                if path == "/api/fill":
                    result = self.service.fill_gap(str(payload.get("context", "")), int(payload.get("n_chars", 0)))
                    self._send_json({"results": result})
                else:
                    result = self.service.detect_errors(
                        str(payload.get("text", "")), str(payload.get("mode", "threshold"))
                    )
                    self._send_json({"results": result})
        except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    service = LogionService()
    RequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    logger.info("Open http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
