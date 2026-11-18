"""Serve the evaluator UI (Laptop 2):  python ui/serve.py [port]
Then open http://localhost:<port> and enter the API URL of Laptop 1,
e.g. http://192.168.x.y:8000
"""
import functools
import http.server
import os
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(os.path.abspath(__file__)))
print(f"PathSnap UI on http://0.0.0.0:{port}")
http.server.ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()
