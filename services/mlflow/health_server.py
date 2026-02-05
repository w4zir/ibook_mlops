from __future__ import annotations

"""
Lightweight health/diagnostics HTTP server for the MLflow container.

Responsibilities:
- Expose a `/healthz` endpoint for container/orchestrator health checks.
- Optionally enforce HTTP Basic Auth via environment variables.
- Optionally probe the MLflow server (default: http://localhost:5000/) and
  return 500 if it is not responding.

This module is intentionally dependency‑free (stdlib only) so it can run in
the slim MLflow image.
"""

import base64
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen


MLFLOW_HEALTH_TARGET = os.getenv("MLFLOW_HEALTH_TARGET", "http://localhost:5000/")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "5001"))
_BASIC_USER = os.getenv("HEALTH_BASIC_AUTH_USER")
_BASIC_PASSWORD = os.getenv("HEALTH_BASIC_AUTH_PASSWORD")


def _check_basic_auth(header_value: Optional[str]) -> bool:
    """
    Validate Authorization header against optional BASIC auth credentials.
    """
    if not _BASIC_USER or not _BASIC_PASSWORD:
        # Auth not configured; always allow.
        return True

    if not header_value or not header_value.startswith("Basic "):
        return False

    try:
        encoded = header_value.split(" ", 1)[1]
        decoded = base64.b64decode(encoded).decode("utf-8")
        user, password = decoded.split(":", 1)
    except Exception:
        return False

    return user == _BASIC_USER and password == _BASIC_PASSWORD


def _probe_mlflow(url: str) -> Tuple[bool, Optional[int]]:
    """
    Best‑effort probe of the MLflow server.

    Returns (ok, status_code).
    """
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=2) as resp:  # nosec B310 - internal call
            return True, resp.status
    except URLError:
        return False, None
    except Exception:
        return False, None


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/healthz":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        if not _check_basic_auth(self.headers.get("Authorization")):
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="mlflow-health"')
            self.end_headers()
            return

        ok, status_code = _probe_mlflow(MLFLOW_HEALTH_TARGET)
        if ok:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            body = f"ok (mlflow_status={status_code})\n".encode("utf-8")
            self.wfile.write(body)
        else:
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.end_headers()
            self.wfile.write(b"mlflow not responding\n")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Reduce noise in container logs; override default stderr logging.
        return


def serve_forever() -> None:
    server = HTTPServer(("", HEALTH_PORT), _Handler)
    server.serve_forever()


def start_in_background() -> threading.Thread:
    """
    Start the health server in a background thread.
    """
    thread = threading.Thread(target=serve_forever, name="health-server", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    serve_forever()

