# Etymon

*The true root hiding beneath the surface.*

A word-connection engine built on GloVe embeddings. Give it two or more words and it finds the word that links them all — the *etymon*, the hidden root beneath them.

Give it a set of words and it surfaces the strongest shared association across all of them — useful for brainstorming, word games, finding a category that covers a list, or any time you want the link hiding beneath a group of words.

Runs two methods in parallel — fast set intersection and deeper best-first graph traversal — then merges and ranks every candidate by its strongest connection.

![Etymon finding the words that connect spring, summer and winter, with the path to each](docs/demo.webp)

## Quick Start

```bash
# Just run it — downloads GloVe and builds the graph automatically on first run
python server.py
# Open http://localhost:8080
```

First run will download GloVe embeddings (~822 MB), extract the 300d file, and build the neighbor graph (~2 min). Subsequent runs load the pre-built graph in seconds.

### Command line usage

**Find the words that connect a set of words** (auto-builds the graph if needed):

```console
$ python graph.py search cat lion

Targets: ['cat', 'lion']
Method: both
Time: 1ms

Results:
  dog                  0.565  [traversal]
    cat: cat → dog
    lion: lion → bear → dog
  cats                 0.558  [traversal]
    cat: cat → cats
    lion: lion → leopard → cats
  elephant             0.515  [traversal]
    cat: cat → monkey → elephant
    lion: lion → elephant
  ...
```

Each result shows the connecting word, its score (the weakest of its links, so higher means it sits close to *every* input), which method found it, and the path the traversal walked from each input.

**Exclude specific words from the answers with `--avoid`** — any word you list here won't be returned as a result (the connecting words still come from the same search; the listed words are just filtered out):

```console
$ python graph.py search cat lion tiger --avoid king

Targets: ['cat', 'lion', 'tiger']
Method: both
Time: 2ms

Results:
  cats                 0.475  [traversal]
  elephant             0.474  [traversal]
  leopard              0.459  [both]
  ...
```

**Explore a single word's nearest neighbors:**

```console
$ python graph.py neighbors engine --n 50

Top 50 neighbors of 'engine':
  engines              0.881
  cylinder             0.591
  diesel               0.589
  horsepower           0.577
  powered              0.567
  turbine              0.555
  ...
```

**Build with custom settings, or point at an existing GloVe file:**

```bash
# Larger vocabulary, more neighbors per word
python graph.py build --vocab 75000 --top-k 200

# Use a GloVe file you already have
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
│  Search engine — runs BOTH methods, then merges        │
│                                                        │
│  A. Set intersection (fast, ~1ms)                      │
│     neighbors(word_A) ∩ neighbors(word_B)              │
│     Progressive widening: top-50 → top-100 → top-150  │
│                                                        │
│  B. Best-first traversal (deep)                        │
│     Walks the graph from each target independently,    │
│     using embedding similarity as heuristic, then      │
│     intersects the reachable sets                      │
│     Depth limit: 2     Node budget: 500 max explored   │
│     Similarity floor: 0.05 minimum                     │
│                                                        │
│  → Candidates from both are merged and ranked by       │
│    strongest connection (weakest-link score). The      │
│    best word wins regardless of which method found it. │
└────────────────────────────────────────────────────────┘
```

## Tuning

All thresholds are configurable. Good starting points:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--vocab` | 50,000 | Dictionary size. 50k covers most common English words. |
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
