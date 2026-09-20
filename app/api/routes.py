"""Minimal standard-library HTTP routes (no web framework required)."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def create_handler(assistant: Any):
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _read_json(self) -> dict:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(body, dict) or not isinstance(body.get("query", ""), str):
                raise ValueError
            return body

        def _send(self, status: int, body: Any, content_type: str = "application/json") -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8") if content_type.startswith("application") else str(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True, "records": len(assistant.documents), "chunks": len(assistant.chunks)})
            elif path in ("/", "/index.html"):
                self._send_file(frontend_dir / "index.html", "text/html")
            elif path == "/static/style.css":
                self._send_file(frontend_dir / "style.css", "text/css")
            elif path == "/static/app.js":
                self._send_file(frontend_dir / "app.js", "text/javascript")
            else:
                self._send(404, {"error": "not_found"})

        def _send_file(self, file_path: Path, content_type: str) -> None:
            try:
                body = file_path.read_bytes()
            except OSError:
                self._send(404, {"error": "not_found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in ("/query", "/query/stream"):
                self._send(404, {"error": "not_found"})
                return
            try:
                body = self._read_json()
                result = assistant.answer(
                    body.get("query", ""),
                    conversation_id=str(body.get("conversation_id", "default")),
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
                )
                if path == "/query/stream":
                    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    # The answer is already grounded and validated by the normal
                    # pipeline; emit it in small chunks for a native chat UX.
                    words = result["answer"].split(" ")
                    for index, word in enumerate(words):
                        token = word if index == 0 else " " + word
                        event = json.dumps({"type": "token", "text": token}, ensure_ascii=False)
                        self.wfile.write(("data: " + event + "\n\n").encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(0.018)
                    self.wfile.write(("data: " + json.dumps(
                        {"type": "done", "result": result}, ensure_ascii=False
                    ) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    return
                self._send(200, result)
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "expected JSON with a string query"})

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler
