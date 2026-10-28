"""Serve the local pixel/data review workspace on loopback only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

import cv2

from .review_store import ConflictError, ReviewStore


def create_server(store: ReviewStore, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def local_request(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != origin.removeprefix("http://") or self.headers.get("Origin", origin) != origin:
                self.send(403, {"error": "仅允许从本地复核页面访问"})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path in {"/", "/app.js", "/style.css"}:
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    path = Path(__file__).parent / "review_ui" / name
                    self.send(200, path.read_bytes(), mimetypes.guess_type(path)[0] + "; charset=utf-8")
                elif parsed.path == "/api/items":
                    self.send(200, {"items": store.list_items(query.get("kind", ["video"])[0], query.get("filter", ["all"])[0])})
                elif parsed.path == "/api/item":
                    self.send(200, store.item(query["id"][0]))
                elif parsed.path == "/asset":
                    path = store.asset(query["id"][0], query["type"][0])
                    self.send(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream")
                elif parsed.path == "/context":
                    self.stream_context(store.asset(query["id"][0], "preview"))
                else:
                    self.send(404, {"error": "页面不存在"})
            except (KeyError, ValueError, OSError) as error:
                self.send(400, {"error": str(error)})

        def stream_context(self, path):
            # Existing MP4V review clips are not decoded by every browser.
            # Stream JPEG frames locally without changing the source exports.
            cap = cv2.VideoCapture(str(path))
            try:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("上下文视频无法解码")
                fps = cap.get(cv2.CAP_PROP_FPS)
                interval = 1 / fps if 0 < fps <= 120 else 1 / 30
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                while ok:
                    started = time.monotonic()
                    encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not encoded:
                        break
                    data = jpeg.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    self.wfile.flush()
                    time.sleep(max(0, interval - (time.monotonic() - started)))
                    ok, frame = cap.read()
                self.wfile.write(b"--frame--\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                cap.release()

        def do_POST(self):
            if not self.local_request():
                return
            if urlparse(self.path).path != "/api/save":
                self.send(404, {"error": "页面不存在"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("请求格式或大小错误")
                payload = json.loads(self.rfile.read(length))
                self.send(200, store.save(payload["id"], payload))
            except ConflictError as error:
                self.send(409, {"error": str(error)})
            except (KeyError, ValueError, TypeError, OSError) as error:
                self.send(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="runs/review_sets/user_events_20261027_v3/pixel_review.template.json")
    parser.add_argument("--dataset", default="datasets/ureter_semis_v2_review_20261027/manifest.json")
    parser.add_argument("--session", default="runs/annotation_sessions/review_20261027")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = ReviewStore(args.template, args.dataset, args.session)
    server = create_server(store, args.port)
    print(f"Local review workspace: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Saved revisions and exports: {store.directory}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
