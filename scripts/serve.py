#!/usr/bin/env python3
"""
Dev server สำหรับ dashboard
- Serves reports/static/ ที่ /
- Serves data/ ที่ /data/
- GET /api/run-online-learning → SSE stream ผลการรัน script
"""
import sys, json, subprocess
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT   = Path(__file__).resolve().parents[1]
STATIC = ROOT / "reports" / "static"
DATA   = ROOT / "data"

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        if self.path == "/api/run-online-learning":
            self._run_sse()
        elif self.path.startswith("/data/"):
            self._serve_data(self.path[6:])
        else:
            super().do_GET()

    def _serve_data(self, filename):
        fpath = DATA / filename
        if not fpath.exists():
            self.send_error(404); return
        data = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _run_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control",  "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        script = str(ROOT / "scripts" / "run_online_learning.py")
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=str(ROOT),
        )
        try:
            for line in proc.stdout:
                stripped = line.rstrip()
                try:
                    json.loads(stripped)   # already valid JSON — pass through
                    payload = stripped
                except (json.JSONDecodeError, ValueError):
                    payload = json.dumps(stripped)   # plain text — encode as string
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            proc.wait()
            self.wfile.write(
                f"data: {{\"__done__\": {proc.returncode}}}\n\n".encode()
            )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    httpd = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"🌐  http://localhost:{port}/online_learning.html")
    print("    Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
