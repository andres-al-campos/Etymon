# Etymon

*The true root hiding beneath the surface.*

A word association engine for Codenames, built on GloVe embeddings. Given 2+ board words, finds the clue word that connects them all — the *etymon*, the hidden link.

Uses set intersection for easy pairs and best-first graph traversal for harder ones.

## Quick Start

```bash
# Just run it — downloads GloVe and builds the graph automatically on first run
python server.py
# Open http://localhost:8080
```

First run will download GloVe embeddings (~822 MB), extract the 300d file, and build the neighbor graph (~2 min). Subsequent runs load the pre-built graph in seconds.

### Command line usage

```bash
# Find clues connecting words (auto-builds graph if needed)
python graph.py search cat lion

# Find clues for 3 words, avoiding "king"
python graph.py search cat lion tiger --avoid king

# Explore a word's neighbors
python graph.py neighbors engine --n 50

# Manually build with custom settings
python graph.py build --vocab 75000 --top-k 200

# Or point to an existing GloVe file
python graph.py build ~/downloads/glove.6B.300d.txt
```

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  GloVe embeddings (50k words × 300 dimensions)        │
│  Auto-downloaded on first run from Stanford NLP        │
└──────────────────┬─────────────────────────────────────┘
                   │ build step (~2 min, one time)
                   ▼
┌────────────────────────────────────────────────────────┐
│  Neighbor graph (50k words × 150 neighbors each)       │
│  Stored as numpy arrays (~60 MB on disk)               │
└──────────────────┬─────────────────────────────────────┘
                   │ query time
                   ▼
┌────────────────────────────────────────────────────────┐
│  Search engine                                         │
│                                                        │
│  1. Try set intersection (fast, ~1ms)                  │
│     neighbors(word_A) ∩ neighbors(word_B)              │
│     Progressive widening: top-50 → top-100 → top-150  │
│                                                        │
│  2. If empty, fall back to best-first traversal        │
│     Uses embedding similarity as search heuristic      │
│     Depth limit: 2 (optionally 3 for high scores)     │
│     Node budget: 500 max explored                      │
│     Similarity floor: 0.05 minimum                     │
└────────────────────────────────────────────────────────┘
```

## Tuning

All thresholds are configurable. Good starting points:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--vocab` | 50,000 | Dictionary size. 50k covers all normal Codenames clues. |
| `--top-k` | 150 | Neighbors per word. Higher = more creative leaps, more noise. |
| `max_depth` | 2 | Graph traversal depth. 2 is usually enough; 3 for desperate cases. |
| `max_nodes` | 500 | Safety valve on traversal. Prevents runaway searches. |
| `min_similarity` | 0.05 | Don't explore branches below this similarity. Prunes dead ends. |

## File Structure

```
Etymon/
├── graph.py      # Core engine: loading, building, searching
├── server.py     # Web server with JSON API
├── ui.html       # Browser UI
├── README.md     # This file
└── graph_data/   # Built graph (auto-created on first run)
    ├── words.json
    ├── embeddings.npy
    ├── neighbor_indices.npy
    ├── neighbor_scores.npy
    └── meta.json
```
