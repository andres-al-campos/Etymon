"""
Etymon — Web Server

A lightweight server using only stdlib.
Serves the UI and provides a JSON API for searches.

Auto-downloads GloVe and builds the graph on first run if needed.

Endpoints:
  GET  /                → Serves the HTML UI
  GET  /api/search      → Find connecting words (params: targets, avoid)
  GET  /api/neighbors   → Get neighbors of a word (params: word, n)
  GET  /api/similarity  → Get similarity between two words (params: a, b)
  GET  /api/stats       → Graph stats (vocab size, etc.)
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from graph import ensure_graph


# Global graph instance
graph = None


class SolverHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves both the UI and the API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self.serve_file("ui.html", "text/html")
        elif path == "/api/search":
            self.handle_search(params)
        elif path == "/api/neighbors":
            self.handle_neighbors(params)
        elif path == "/api/similarity":
            self.handle_similarity(params)
        elif path == "/api/stats":
            self.handle_stats()
        else:
            self.send_error(404)

    def serve_file(self, filename, content_type):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {filename}")

    def json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def handle_search(self, params):
        targets = params.get("targets", [""])[0].split(",")
        targets = [t.strip() for t in targets if t.strip()]
        avoid = params.get("avoid", [""])[0].split(",")
        avoid = [a.strip() for a in avoid if a.strip()]
        limit = int(params.get("limit", ["20"])[0])

        if len(targets) < 2:
            self.json_response({"error": "Need at least 2 target words"}, 400)
            return

        result = graph.find_connections(targets, avoid_words=avoid or None, result_limit=limit)
        self.json_response(result)

    def handle_neighbors(self, params):
        word = params.get("word", [""])[0].strip().lower()
        n = int(params.get("n", ["30"])[0])

        if not word:
            self.json_response({"error": "Missing 'word' parameter"}, 400)
            return

        neighbors = graph.get_neighbors(word, n=n)
        self.json_response({
            "word": word,
            "neighbors": [{"word": w, "score": round(s, 4)} for w, s in neighbors],
        })

    def handle_similarity(self, params):
        a = params.get("a", [""])[0].strip().lower()
        b = params.get("b", [""])[0].strip().lower()

        if not a or not b:
            self.json_response({"error": "Need both 'a' and 'b' parameters"}, 400)
            return

        sim = graph.cosine_similarity_word(a, b)
        if sim is None:
            self.json_response({"error": f"Word not found in vocabulary"}, 404)
            return

        self.json_response({"a": a, "b": b, "similarity": round(sim, 4)})

    def handle_stats(self):
        self.json_response({
            "vocab_size": graph.vocab_size,
            "top_k": graph.top_k,
            "sample_words": graph.words[:20],
        })

    # Suppress default logging
    def log_message(self, format, *args):
        print(f"  {args[0]}")


def main():
    global graph

    graph_path = sys.argv[1] if len(sys.argv) > 1 else "./graph_data"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    # Auto-downloads GloVe and builds graph if needed
    graph = ensure_graph(graph_path)

    server = HTTPServer(("0.0.0.0", port), SolverHandler)
    print(f"\nEtymon running at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
