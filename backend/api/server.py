import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import mimetypes
import traceback
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

_engine_instance = None

#gets the inference engine required to process predicition request
def get_engine():
    global _engine_instance
    if _engine_instance is None:
        from backend.api.inference_engine import engine
        _engine_instance = engine
    return _engine_instance

PORT = 8000
FRONTEND_DIR = os.path.abspath("frontend")

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.json")
try:
    with open(REGISTRY_PATH, "r") as f:
        MODEL_REGISTRY = json.load(f)
except Exception:
    MODEL_REGISTRY = {}

#allows the server to handle more than one request at a time
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests concurrently in separate threads."""
    daemon_threads = True

#handles the request coming from the frontend and send suitable responses
class FakeLensRequestHandler(SimpleHTTPRequestHandler):
    """
    HTTP handler serving API endpoints and frontend assets.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    # Sets the response headers
    def _set_json_headers(self, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # Handles the browser preflight request before actual api request
    def do_OPTIONS(self):
        self._set_json_headers(HTTPStatus.NO_CONTENT)

    # Handles the request for model information, evaluation and other resiurces.
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            eng = get_engine()
            data = {
                "status": "healthy" if eng.is_ready else "degraded",
                "engine_ready": eng.is_ready,
                "device": str(eng.device),
                "datasets": {
                    "deepfakeface": {
                        "name": "DeepFakeFace",
                        "description": "Facial Forgery Detection & Attribution (V4 60K benchmark)",
                        "domain": "Human Facial Portraits",
                        "models_count": 10
                    },
                    "artifact": {
                        "name": "ArtiFact",
                        "description": "General Image Forgery Detection & Attribution (55K benchmark)",
                        "domain": "General Objects & Natural Imagery",
                        "models_count": 10
                    },
                    "cifake": {
                        "name": "CIFAKE",
                        "description": "CIFAR-10 Real vs AI Generated Image Benchmark",
                        "domain": "32x32 RGB Natural vs Diffusion Synthesis",
                        "models_count": 4
                    }
                },
                "total_registered_models": 24,
                "default_dataset": "deepfakeface",
                "default_family": "efficientnet_b0"
            }
            self._set_json_headers(HTTPStatus.OK)
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path == "/api/models":
            self._set_json_headers(HTTPStatus.OK)
            self.wfile.write(json.dumps(MODEL_REGISTRY).encode("utf-8"))
            return

        elif path == "/api/benchmarks" or path == "/api/metrics":

            self._set_json_headers(HTTPStatus.OK)
            self.wfile.write(json.dumps(MODEL_REGISTRY).encode("utf-8"))
            return

        if path == "/" or not os.path.exists(os.path.join(FRONTEND_DIR, path.lstrip("/"))):
            self.path = "/index.html"

        return super().do_GET()

    # receives the uploaded image
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length <= 0:
                self._set_json_headers(HTTPStatus.BAD_REQUEST)
                self.wfile.write(json.dumps({"error": "Empty request body"}).encode("utf-8"))
                return

            if content_length > 25 * 1024 * 1024:  # 25MB max
                self._set_json_headers(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                self.wfile.write(json.dumps({"error": "File size exceeds 25MB maximum limit."}).encode("utf-8"))
                return

            raw_body = self.rfile.read(content_length)

            image_bytes = None
            dataset = "deepfakeface"
            model_family = "efficientnet_b0"

            if "multipart/form-data" in content_type:
                try:
                    boundary = content_type.split("boundary=")[1].encode()
                    parts = raw_body.split(b"--" + boundary)
                    for part in parts:
                        if b'name="dataset"' in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                ds_val = part[header_end + 4:].rstrip(b"\r\n--").decode("utf-8").strip()
                                if ds_val:
                                    dataset = ds_val
                        elif b'name="model_family"' in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                family_val = part[header_end + 4:].rstrip(b"\r\n--").decode("utf-8").strip()
                                if family_val:
                                    model_family = family_val
                        elif b"filename=" in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                image_bytes = part[header_end + 4:].rstrip(b"\r\n--")
                except Exception as e:
                    self._set_json_headers(HTTPStatus.BAD_REQUEST)
                    self.wfile.write(json.dumps({"error": f"Failed to parse multipart upload: {str(e)}"}).encode("utf-8"))
                    return
            else:
                image_bytes = raw_body

            if not image_bytes:
                self._set_json_headers(HTTPStatus.BAD_REQUEST)
                self.wfile.write(json.dumps({"error": "No image payload found in request."}).encode("utf-8"))
                return

            try:
                eng = get_engine()
                result = eng.analyze_image(image_bytes, dataset=dataset, model_family=model_family)
                self._set_json_headers(HTTPStatus.OK)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except ValueError as ve:
                self._set_json_headers(HTTPStatus.BAD_REQUEST)
                self.wfile.write(json.dumps({"error": str(ve)}).encode("utf-8"))
            except Exception as e:
                traceback.print_exc()
                self._set_json_headers(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.wfile.write(json.dumps({"error": f"Inference engine failure: {str(e)}"}).encode("utf-8"))
            return

        self._set_json_headers(HTTPStatus.NOT_FOUND)
        self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))


# starts the application server using the given host address and port
def run_server(port=PORT, bind="127.0.0.1"):
    server_address = (bind, port)
    httpd = ThreadedHTTPServer(server_address, FakeLensRequestHandler)
    print(f"\n=======================================================")
    print(f" FakeLens Unified Web Server: http://{bind}:{port}")
    print(f" Datasets: DeepFakeFace, ArtiFact & CIFAKE (24 Models)")
    print(f" Press Ctrl+C to terminate.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down FakeLens server...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FakeLens Web & API Server")
    parser.add_argument("--port", "-p", type=int, default=PORT, help="Port to listen on (default: 8000)")
    parser.add_argument("--bind", "-b", type=str, default="127.0.0.1", help="Address to bind (default: 127.0.0.1)")
    args, unknown = parser.parse_known_args()

    port = args.port
    if unknown and unknown[0].isdigit():
        port = int(unknown[0])

    run_server(port=port, bind=args.bind)
