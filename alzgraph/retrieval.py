"""Graph-RAG retrieval over AlzKG.

The retriever matches the paper's Graph-RAG setting: it seeds a personalized
PageRank (PPR) walk from query-mentioned entities, keeps the highest-scoring
neighborhood, and serializes multi-hop reasoning paths (with per-edge evidence
counts) that are handed to the language model.

It uses ``networkx`` when available for the PPR step, and otherwise falls back to
a self-contained power-iteration implementation so the release runs with no hard
graph dependency.
"""

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Tuple

from .common import normalize_text, read_json

try:  # optional fast path
    import networkx as nx

    _HAS_NX = True
except Exception:  # pragma: no cover - exercised only when networkx is absent
    _HAS_NX = False


class AlzGraphRetriever:
    """PPR-style graph retriever matching the paper's Graph-RAG setting."""

    def __init__(
        self,
        triplets_path: str,
        ppr_alpha: float = 0.15,
        max_subgraph_nodes: int = 30,
        max_paths: int = 12,
        max_depth: int = 4,
    ) -> None:
        self.triplets = read_json(triplets_path)
        self.ppr_alpha = ppr_alpha
        self.max_subgraph_nodes = max_subgraph_nodes
        self.max_paths = max_paths
        self.max_depth = max_depth
        self.entity_names: Dict[str, str] = {}
        self.out_edges: Dict[str, List[dict]] = defaultdict(list)
        self.nodes: set[str] = set()
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        for row in self.triplets:
            head = normalize_text(row.get("head", "")).lower()
            tail = normalize_text(row.get("tail", "")).lower()
            if not head or not tail:
                continue
            self.entity_names.setdefault(head, row.get("head", head))
            self.entity_names.setdefault(tail, row.get("tail", tail))
            self.nodes.add(head)
            self.nodes.add(tail)
            weight = max(float(row.get("paper_count", 1)), 1.0)
            self.out_edges[head].append(
                {
                    "tail": tail,
                    "relation": row.get("relation", "related_to"),
                    "weight": weight,
                    "paper_count": row.get("paper_count", 1),
                    "weight_label": row.get("weight_label", "papers"),
                    "evidence": row.get("evidence", row.get("paper_ids", [])),
                }
            )

    # --------------------------------------------------------------- retrieve
    def retrieve(self, query: str) -> Dict[str, object]:
        seeds = self.match_entities(query)
        if not seeds:
            return {"seeds": [], "paths": [], "triplets": []}
        scores = self._pagerank(seeds)
        keep = {
            node
            for node, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                : self.max_subgraph_nodes
            ]
        }
        keep.update(seeds)
        paths = self.serialize_paths(keep, seeds)
        return {
            "seeds": [self.entity_names.get(s, s) for s in seeds],
            "paths": paths,
            "triplets": self._triplets_from_subgraph(keep),
        }

    def match_entities(self, query: str) -> List[str]:
        q = f" {query.lower()} "
        hits = []
        for entity in self.nodes:
            if len(entity) < 3:
                continue
            if f" {entity} " in q or entity.replace("-", " ") in q:
                hits.append(entity)
        # Prefer longer (more specific) matches first.
        return sorted(set(hits), key=len, reverse=True)[:8]

    # --------------------------------------------------------------- pagerank
    def _pagerank(self, seeds: Iterable[str]) -> Dict[str, float]:
        seeds = list(seeds)
        if _HAS_NX:
            return self._pagerank_nx(seeds)
        return self._pagerank_python(seeds)

    def _pagerank_nx(self, seeds: List[str]) -> Dict[str, float]:
        graph = nx.DiGraph()
        for head, edges in self.out_edges.items():
            for e in edges:
                graph.add_edge(head, e["tail"], weight=e["weight"])
        for node in self.nodes:
            graph.add_node(node)
        personalization = {node: (1.0 if node in set(seeds) else 0.0) for node in graph.nodes}
        if sum(personalization.values()) == 0:
            return {}
        return nx.pagerank(
            graph,
            alpha=1 - self.ppr_alpha,
            personalization=personalization,
            weight="weight",
            max_iter=200,
            tol=1e-8,
        )

    def _pagerank_python(self, seeds: List[str], iters: int = 100, tol: float = 1e-9) -> Dict[str, float]:
        damping = 1.0 - self.ppr_alpha  # follow-edge probability
        restart = self.ppr_alpha        # teleport-to-seeds probability
        seed_set = set(seeds)
        n_seeds = max(len(seed_set), 1)
        personalization = {node: (1.0 / n_seeds if node in seed_set else 0.0) for node in self.nodes}

        # Pre-normalize out-edge weights.
        out_norm: Dict[str, List[Tuple[str, float]]] = {}
        for node in self.nodes:
            edges = self.out_edges.get(node, [])
            total = sum(e["weight"] for e in edges)
            if total > 0:
                out_norm[node] = [(e["tail"], e["weight"] / total) for e in edges]
            else:
                out_norm[node] = []

        score = dict(personalization)
        for _ in range(iters):
            nxt = {node: restart * personalization[node] for node in self.nodes}
            dangling_mass = 0.0
            for node in self.nodes:
                s = score[node]
                if not out_norm[node]:
                    dangling_mass += s
                    continue
                for tail, w in out_norm[node]:
                    nxt[tail] += damping * s * w
            # Dangling mass teleports back to the personalization vector.
            if dangling_mass:
                for node in self.nodes:
                    nxt[node] += damping * dangling_mass * personalization[node]
            delta = sum(abs(nxt[n] - score[n]) for n in self.nodes)
            score = nxt
            if delta < tol:
                break
        return score

    # ------------------------------------------------------------------ paths
    def serialize_paths(self, keep: set, seeds: Iterable[str]) -> List[str]:
        keep = set(keep)
        paths: List[Tuple[float, str]] = []
        for seed in seeds:
            if seed not in keep:
                continue
            queue = deque([(seed, [seed], 0)])
            while queue:
                node, nodes, depth = queue.popleft()
                if depth >= self.max_depth:
                    continue
                for e in self.out_edges.get(node, []):
                    nxt = e["tail"]
                    if nxt not in keep or nxt in nodes:
                        continue
                    new_nodes = nodes + [nxt]
                    text = self._format_path(new_nodes)
                    score = self._path_score(new_nodes)
                    paths.append((score, text))
                    queue.append((nxt, new_nodes, depth + 1))
        dedup: Dict[str, float] = {}
        for score, text in paths:
            dedup[text] = max(score, dedup.get(text, 0))
        return [
            text
            for text, _ in sorted(dedup.items(), key=lambda item: item[1], reverse=True)[
                : self.max_paths
            ]
        ]

    def _edge(self, head: str, tail: str) -> dict | None:
        for e in self.out_edges.get(head, []):
            if e["tail"] == tail:
                return e
        return None

    def _path_score(self, nodes: List[str]) -> float:
        total = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            e = self._edge(a, b)
            total += e.get("paper_count", 1) if e else 0
        return total

    def _format_path(self, nodes: List[str]) -> str:
        pieces = [self.entity_names.get(nodes[0], nodes[0])]
        for a, b in zip(nodes[:-1], nodes[1:]):
            e = self._edge(a, b) or {}
            rel = e.get("relation", "related_to")
            pc = e.get("paper_count", 1)
            label = e.get("weight_label", "papers")
            pieces.append(f"--{rel} [{pc} {label}]--> {self.entity_names.get(b, b)}")
        return " ".join(pieces)

    def _triplets_from_subgraph(self, keep: set) -> List[dict]:
        keep = set(keep)
        rows = []
        for head in keep:
            for e in self.out_edges.get(head, []):
                if e["tail"] in keep:
                    rows.append(
                        {
                            "head": self.entity_names.get(head, head),
                            "relation": e["relation"],
                            "tail": self.entity_names.get(e["tail"], e["tail"]),
                            "paper_count": e.get("paper_count", 1),
                        }
                    )
        return rows
