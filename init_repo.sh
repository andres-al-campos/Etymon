#!/bin/bash
# Initialize Etymon: git repo + uv environment
cd "$(dirname "$0")"

# Git
git init
git add -A
git commit -m "Initial commit: Etymon — Codenames solver

The true root hiding beneath the surface.

- graph.py: Core engine (load GloVe, build neighbor graph, search)
- server.py: Lightweight web server with JSON API
- ui.html: Dark-themed browser UI
- Search: set intersection (fast) with best-first traversal fallback
- Progressive widening: top-50 → top-100 → top-150 neighbors
- Configurable: vocab size, neighbor count, depth, node budget"

# uv
uv sync

echo ""
echo "Etymon initialized."
echo ""
echo "Next steps:"
echo "  1. wget https://nlp.stanford.edu/data/glove.6B.zip"
echo "  2. unzip glove.6B.zip"
echo "  3. uv run python graph.py build glove.6B.300d.txt"
echo "  4. uv run python server.py"
echo "  5. Open http://localhost:8080"
