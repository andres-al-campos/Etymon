"""
Etymon — Word-Connection Engine

Knowledge Graph Builder & Search Engine.

Builds a dense word association graph from GloVe embeddings and provides
best-first search to find words that connect a set of input words.

Architecture:
  - Load GloVe embeddings (top N most common English words)
  - For each word, pre-compute top-K nearest neighbors by cosine similarity
  - At query time, run BOTH set intersection and best-first traversal
  - Merge and rank all candidates by strongest connection
"""

import numpy as np
import heapq
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

# ─── Constants ────────────────────────────────────────────────────────────

# GloVe source
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
GLOVE_FILE = "glove.6B.300d.txt"
GLOVE_DIMENSIONS = 300           # dimensionality of word vectors

# Graph construction
DEFAULT_VOCAB_SIZE = 50_000      # top N most frequent words to load
DEFAULT_TOP_K = 150              # neighbors per word (Dunbar-ish)
DEFAULT_BATCH_SIZE = 1000        # words processed per batch during build
MIN_WORD_LENGTH = 2              # skip single-character words
NORM_EPSILON = 1e-10             # avoid division by zero in normalization

# Search: set intersection (fast mode)
INTERSECTION_WIDTHS = [50, 100]  # progressive widening steps (top_k appended automatically)
INTERSECTION_MIN_CANDIDATES = 5  # minimum intersection size before widening
DEFAULT_RESULT_LIMIT = 20        # max results returned

# Search: best-first traversal (deep mode)
TRAVERSAL_MAX_DEPTH = 2          # maximum hops from any target word
TRAVERSAL_MAX_NODES = 500        # safety valve: max nodes explored total
TRAVERSAL_MIN_SIMILARITY = 0.05  # don't explore branches below this score
TRAVERSAL_BRANCH_FACTOR = 15     # max neighbors expanded per node

# Paths
DEFAULT_GRAPH_DIR = "./graph_data"


# ─── Download ─────────────────────────────────────────────────────────────

def ensure_glove(glove_path: str = GLOVE_FILE) -> str:
    """
    Check if GloVe embeddings exist, download if not.

    Downloads glove.6B.zip (~822 MB), extracts glove.6B.300d.txt,
    and cleans up the zip and unused dimension files.

    Returns the path to the .txt file.
    """
    if os.path.exists(glove_path):
        return glove_path

    glove_dir = os.path.dirname(glove_path) or "."
    txt_path = os.path.join(glove_dir, GLOVE_FILE)

    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(glove_dir, GLOVE_ZIP)

    # Download
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe embeddings (~822 MB)...")
        print(f"  From: {GLOVE_URL}")
        print(f"  To:   {zip_path}")

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"\r  {pct:.0f}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

        urlretrieve(GLOVE_URL, zip_path, reporthook=progress_hook)
        print()  # newline after progress

    # Extract just the 300d file
    print(f"Extracting {GLOVE_FILE}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_FILE, glove_dir)

    # Clean up: remove zip and unused dimension files
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"  Removed {GLOVE_ZIP}")

    for dim in ["50d", "100d", "200d"]:
        other = os.path.join(glove_dir, f"glove.6B.{dim}.txt")
        if os.path.exists(other):
            os.remove(other)

    print(f"  Ready: {txt_path}")
    return txt_path


class WordGraph:
    """
    Dense word association graph built from GloVe embeddings.

    The graph stores two things:
      1. Raw embedding matrix (vocab_size × 300) for on-demand similarity
      2. Pre-computed neighbor lists (vocab_size × top_k) for fast lookup

    The search has two modes:
      - Fast mode: set intersection of pre-computed neighbor lists
      - Deep mode: best-first traversal using embeddings as heuristic
    """

    def __init__(self):
        self.words: list[str] = []
        self.word_to_idx: dict[str, int] = {}
        self.embeddings: np.ndarray = None
        self.norms: np.ndarray = None
        self.neighbors: dict[int, list[tuple[int, float]]] = {}
        self.vocab_size: int = 0
        self.top_k: int = DEFAULT_TOP_K

    # ─── Loading ────────────────────────────────────────────────────────

    def load_glove(self, glove_path: str, max_words: int = DEFAULT_VOCAB_SIZE):
        """
        Load GloVe embeddings from a text file.

        GloVe files are pre-sorted by frequency, so loading the first
        max_words lines gives you the most common English words.
        """
        print(f"Loading GloVe embeddings (top {max_words:,} words)...")
        start = time.time()

        words = []
        vectors = []

        with open(glove_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_words:
                    break

                parts = line.rstrip().split(" ")
                word = parts[0]

                if not word.isalpha() or len(word) < MIN_WORD_LENGTH:
                    continue

                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                except ValueError:
                    continue

                if vector.shape[0] != GLOVE_DIMENSIONS:
                    continue

                words.append(word)
                vectors.append(vector)

        self.words = words
        self.word_to_idx = {w: i for i, w in enumerate(words)}
        self.embeddings = np.array(vectors, dtype=np.float32)
        self.vocab_size = len(words)

        # Precompute norms — shape: (vocab_size,) flat
        self.norms = np.linalg.norm(self.embeddings, axis=1)
        self.norms = np.maximum(self.norms, NORM_EPSILON)

        elapsed = time.time() - start
        print(f"Loaded {self.vocab_size:,} words in {elapsed:.1f}s")

    # ─── Graph Building ─────────────────────────────────────────────────

    def build_neighbor_graph(self, top_k: int = DEFAULT_TOP_K, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Pre-compute top-K nearest neighbors for every word.
        Processes in batches to avoid memory issues.
        """
        self.top_k = top_k
        print(f"Building neighbor graph (top {top_k} per word)...")
        start = time.time()

        normed = self.embeddings / self.norms[:, np.newaxis]
        total_batches = (self.vocab_size + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.vocab_size)
            batch = normed[batch_start:batch_end]

            sims = batch @ normed.T

            for i in range(batch_end - batch_start):
                word_idx = batch_start + i
                row = sims[i]
                row[word_idx] = -1.0

                top_indices = np.argpartition(row, -top_k)[-top_k:]
                top_scores = row[top_indices]

                sorted_order = np.argsort(-top_scores)
                top_indices = top_indices[sorted_order]
                top_scores = top_scores[sorted_order]

                self.neighbors[word_idx] = [
                    (int(idx), float(score))
                    for idx, score in zip(top_indices, top_scores)
                ]

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"  {progress:.0f}% ({batch_end:,}/{self.vocab_size:,} words)")

        elapsed = time.time() - start
        print(f"Graph built in {elapsed:.1f}s")

    # ─── Saving / Loading ───────────────────────────────────────────────

    def save(self, path: str):
        """Save the full graph to disk (embeddings + neighbor lists)."""
        print(f"Saving graph to {path}...")
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "words.json"), "w") as f:
            json.dump(self.words, f)

        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)

        n_entries = self.vocab_size * self.top_k
        indices = np.zeros(n_entries, dtype=np.int32)
        scores = np.zeros(n_entries, dtype=np.float32)

        for word_idx, neighbor_list in self.neighbors.items():
            offset = word_idx * self.top_k
            for j, (nidx, nscore) in enumerate(neighbor_list):
                indices[offset + j] = nidx
                scores[offset + j] = nscore

        np.save(os.path.join(path, "neighbor_indices.npy"), indices)
        np.save(os.path.join(path, "neighbor_scores.npy"), scores)

        meta = {"vocab_size": self.vocab_size, "top_k": self.top_k}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        total_size = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
        )
        print(f"Saved ({total_size / 1024 / 1024:.1f} MB)")

    def load(self, path: str):
        """Load a pre-built graph from disk."""
        print(f"Loading graph from {path}...")
        start = time.time()

        with open(os.path.join(path, "words.json"), "r") as f:
            self.words = json.load(f)

        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        self.vocab_size = len(self.words)

        self.norms = np.linalg.norm(self.embeddings, axis=1)
        self.norms = np.maximum(self.norms, NORM_EPSILON)

        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        self.top_k = meta["top_k"]

        indices = np.load(os.path.join(path, "neighbor_indices.npy"))
        scores_arr = np.load(os.path.join(path, "neighbor_scores.npy"))

        self.neighbors = {}
        for word_idx in range(self.vocab_size):
            offset = word_idx * self.top_k
            self.neighbors[word_idx] = [
                (int(indices[offset + j]), float(scores_arr[offset + j]))
                for j in range(self.top_k)
                if scores_arr[offset + j] > 0
            ]

        elapsed = time.time() - start
        print(f"Loaded {self.vocab_size:,} words, {self.top_k} neighbors each ({elapsed:.1f}s)")

    # ─── Similarity ─────────────────────────────────────────────────────

    def cosine_similarity(self, idx_a: int, idx_b: int) -> float:
        """Cosine similarity between two words by index."""
        a = self.embeddings[idx_a]
        b = self.embeddings[idx_b]
        return float(np.dot(a, b) / (self.norms[idx_a] * self.norms[idx_b]))

    def cosine_similarity_word(self, word_a: str, word_b: str) -> Optional[float]:
        """Cosine similarity between two words by string."""
        if word_a not in self.word_to_idx or word_b not in self.word_to_idx:
            return None
        return self.cosine_similarity(
            self.word_to_idx[word_a],
            self.word_to_idx[word_b]
        )

    def get_neighbors(self, word: str, n: int = 20) -> list[tuple[str, float]]:
        """Get top-N neighbors of a word from the pre-computed graph."""
        if word not in self.word_to_idx:
            return []
        idx = self.word_to_idx[word]
        if idx not in self.neighbors:
            return []
        return [
            (self.words[nidx], score)
            for nidx, score in self.neighbors[idx][:n]
        ]

    # ─── Search: Set Intersection (Fast Mode) ───────────────────────────

    def find_connections_intersection(
        self,
        target_words: list[str],
        avoid_words: list[str] = None,
        top_n: int = DEFAULT_TOP_K,
        result_limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[tuple[str, float]]:
        """
        Find connecting words by intersecting neighbor lists.
        Uses progressive widening: starts with top-50, widens to top-N.
        Scores by minimum similarity across all targets (weakest link).
        """
        target_indices = []
        for w in target_words:
            w_lower = w.lower()
            if w_lower not in self.word_to_idx:
                print(f"  Warning: '{w}' not in vocabulary, skipping")
                continue
            target_indices.append(self.word_to_idx[w_lower])

        if len(target_indices) < 2:
            return []

        avoid_indices = set()
        if avoid_words:
            for w in avoid_words:
                w_lower = w.lower()
                if w_lower in self.word_to_idx:
                    avoid_indices.add(self.word_to_idx[w_lower])

        excluded = set(target_indices) | avoid_indices

        widths = INTERSECTION_WIDTHS + [top_n]
        candidates = []

        for width in widths:
            neighbor_sets = []
            neighbor_scores = {}

            for tidx in target_indices:
                neighbors = self.neighbors.get(tidx, [])[:width]
                nset = set()
                for nidx, nscore in neighbors:
                    if nidx not in excluded:
                        nset.add(nidx)
                        neighbor_scores[(nidx, tidx)] = nscore
                neighbor_sets.append(nset)

            intersection = neighbor_sets[0]
            for ns in neighbor_sets[1:]:
                intersection = intersection & ns

            if len(intersection) >= INTERSECTION_MIN_CANDIDATES:
                for idx in intersection:
                    min_score = min(
                        neighbor_scores.get((idx, tidx), 0.0)
                        for tidx in target_indices
                    )
                    candidates.append((idx, min_score))
                break

        if not candidates:
            return []

        candidates.sort(key=lambda x: -x[1])
        return [
            (self.words[idx], score)
            for idx, score in candidates[:result_limit]
        ]

    # ─── Search: Best-First Traversal (Deep Mode) ───────────────────────

    def find_connections_traversal(
        self,
        target_words: list[str],
        avoid_words: list[str] = None,
        max_depth: int = TRAVERSAL_MAX_DEPTH,
        max_nodes: int = TRAVERSAL_MAX_NODES,
        min_similarity: float = TRAVERSAL_MIN_SIMILARITY,
        branch_factor: int = TRAVERSAL_BRANCH_FACTOR,
        result_limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[dict]:
        """
        Find connecting words via best-first graph traversal.
        Walks the graph from each target word independently, using
        embedding similarity as heuristic, then intersects reachable sets.
        """
        target_indices = []
        target_strs = []
        for w in target_words:
            w_lower = w.lower()
            if w_lower in self.word_to_idx:
                target_indices.append(self.word_to_idx[w_lower])
                target_strs.append(w_lower)

        if len(target_indices) < 2:
            return []

        avoid_indices = set()
        if avoid_words:
            for w in avoid_words:
                w_lower = w.lower()
                if w_lower in self.word_to_idx:
                    avoid_indices.add(self.word_to_idx[w_lower])

        excluded = set(target_indices) | avoid_indices
        reachable = {}

        for ti, tidx in enumerate(target_indices):
            reached = {}
            visited = set()
            nodes_explored = 0

            initial_neighbors = self.neighbors.get(tidx, [])[:branch_factor]
            heap = []
            for nidx, nscore in initial_neighbors:
                if nidx not in excluded and nscore >= min_similarity:
                    heapq.heappush(heap, (-nscore, nidx, 1, [target_strs[ti], self.words[nidx]]))

            while heap and nodes_explored < max_nodes:
                neg_score, current_idx, depth, path = heapq.heappop(heap)
                score = -neg_score

                if current_idx in visited:
                    continue
                visited.add(current_idx)
                nodes_explored += 1

                if current_idx not in reached or score > reached[current_idx][0]:
                    reached[current_idx] = (score, path)

                if depth < max_depth:
                    for nidx, nscore in self.neighbors.get(current_idx, [])[:branch_factor]:
                        if nidx not in visited and nidx not in excluded:
                            other_sims = []
                            for oidx in target_indices:
                                if oidx != tidx:
                                    other_sims.append(self.cosine_similarity(nidx, oidx))
                            heuristic = (nscore + sum(other_sims) / max(len(other_sims), 1)) / 2

                            if heuristic >= min_similarity:
                                new_path = path + [self.words[nidx]]
                                heapq.heappush(heap, (-heuristic, nidx, depth + 1, new_path))

            reachable[ti] = reached

        if not reachable:
            return []

        common_words = set(reachable[0].keys())
        for ti in range(1, len(target_indices)):
            common_words &= set(reachable[ti].keys())

        results = []
        for word_idx in common_words:
            if word_idx in excluded:
                continue

            min_score = min(reachable[ti][word_idx][0] for ti in range(len(target_indices)))
            paths = {target_strs[ti]: reachable[ti][word_idx][1] for ti in range(len(target_indices))}

            results.append({
                "word": self.words[word_idx],
                "score": min_score,
                "paths": paths,
            })

        results.sort(key=lambda x: -x["score"])
        return results[:result_limit]

    # ─── Combined Search ────────────────────────────────────────────────

    def find_connections(
        self,
        target_words: list[str],
        avoid_words: list[str] = None,
        result_limit: int = DEFAULT_RESULT_LIMIT,
    ) -> dict:
        """
        Find connecting words by running BOTH intersection and traversal,
        then merging and ranking all candidates together.
        The strongest connection wins regardless of which method found it.
        """
        start = time.time()

        # Run both methods
        intersection_results = self.find_connections_intersection(
            target_words, avoid_words,
            top_n=self.top_k,
            result_limit=result_limit * 2,
        )

        traversal_results = self.find_connections_traversal(
            target_words, avoid_words,
            result_limit=result_limit * 2,
        )

        # Merge into a single dict keyed by word
        merged = {}

        for word, score in intersection_results:
            merged[word] = {
                "word": word,
                "score": score,
                "source": "intersection",
            }

        for r in traversal_results:
            word = r["word"]
            if word not in merged or r["score"] > merged[word]["score"]:
                merged[word] = {
                    "word": word,
                    "score": r["score"],
                    "source": "traversal",
                    "paths": r.get("paths"),
                }
            elif word in merged and "paths" not in merged[word] and r.get("paths"):
                merged[word]["paths"] = r["paths"]
                merged[word]["source"] = "both"

        # Sort by score descending
        results = sorted(merged.values(), key=lambda x: -x["score"])[:result_limit]

        # Determine which methods contributed
        sources = set(r["source"] for r in results)
        if "both" in sources or ("intersection" in sources and "traversal" in sources):
            method = "both"
        elif "traversal" in sources:
            method = "traversal"
        else:
            method = "intersection"

        elapsed = time.time() - start
        return {
            "method": method,
            "results": results,
            "targets": target_words,
            "elapsed_ms": round(elapsed * 1000),
        }


# ─── Auto-setup ───────────────────────────────────────────────────────────

def ensure_graph(graph_dir: str = DEFAULT_GRAPH_DIR,
                 vocab: int = DEFAULT_VOCAB_SIZE,
                 top_k: int = DEFAULT_TOP_K) -> WordGraph:
    """
    Load the graph if it exists, otherwise download GloVe and build it.
    This is the one-call entry point — handles everything automatically.
    """
    g = WordGraph()

    if os.path.exists(os.path.join(graph_dir, "meta.json")):
        g.load(graph_dir)
        return g

    print("Graph not found — building from scratch.\n")
    glove_path = ensure_glove()
    g.load_glove(glove_path, max_words=vocab)
    g.build_neighbor_graph(top_k=top_k)
    g.save(graph_dir)
    return g


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Etymon — Word-Connection Engine")
    sub = parser.add_subparsers(dest="command")

    # Build command
    build_cmd = sub.add_parser("build", help="Build graph from GloVe embeddings")
    build_cmd.add_argument("glove_path", nargs="?", default=None,
                          help="Path to glove.6B.300d.txt (auto-downloads if omitted)")
    build_cmd.add_argument("--vocab", type=int, default=DEFAULT_VOCAB_SIZE, help="Vocabulary size")
    build_cmd.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Neighbors per word")
    build_cmd.add_argument("--output", default=DEFAULT_GRAPH_DIR, help="Output directory")

    # Search command
    search_cmd = sub.add_parser("search", help="Find words that connect the inputs")
    search_cmd.add_argument("targets", nargs="+", help="Words to connect")
    search_cmd.add_argument("--graph", default=DEFAULT_GRAPH_DIR, help="Graph directory")
    search_cmd.add_argument("--avoid", nargs="*",
                            help="Words to exclude from the results (won't be returned as answers)")
    search_cmd.add_argument("--limit", type=int, default=DEFAULT_RESULT_LIMIT, help="Max results")

    # Neighbors command (for exploration)
    nb_cmd = sub.add_parser("neighbors", help="Show neighbors of a word")
    nb_cmd.add_argument("word", help="Word to explore")
    nb_cmd.add_argument("--graph", default=DEFAULT_GRAPH_DIR, help="Graph directory")
    nb_cmd.add_argument("--n", type=int, default=50, help="Number of neighbors")

    args = parser.parse_args()

    if args.command == "build":
        glove_path = args.glove_path or ensure_glove()
        g = WordGraph()
        g.load_glove(glove_path, max_words=args.vocab)
        g.build_neighbor_graph(top_k=args.top_k)
        g.save(args.output)

    elif args.command == "search":
        g = ensure_graph(args.graph)
        result = g.find_connections(args.targets, avoid_words=args.avoid, result_limit=args.limit)

        print(f"\nTargets: {result['targets']}")
        print(f"Method: {result['method']}")
        print(f"Time: {result['elapsed_ms']}ms")
        print(f"\nResults:")
        for r in result["results"]:
            source_tag = f"[{r.get('source', '?')}]"
            print(f"  {r['word']:20s} {r['score']:.3f}  {source_tag}")
            if r.get("paths"):
                for target, path in r["paths"].items():
                    print(f"    {target}: {' → '.join(path)}")

    elif args.command == "neighbors":
        g = ensure_graph(args.graph)
        neighbors = g.get_neighbors(args.word.lower(), n=args.n)
        if neighbors:
            print(f"\nTop {len(neighbors)} neighbors of '{args.word}':")
            for word, score in neighbors:
                print(f"  {word:20s} {score:.3f}")
        else:
            print(f"'{args.word}' not found in vocabulary")

    else:
        parser.print_help()
