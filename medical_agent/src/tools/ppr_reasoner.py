"""
PPR (Personalized PageRank) with IDF edge weighting and type filtering.

IDF formula: idf(v) = log(1 + N / (1 + degree(v)))
Edge weight = base_weight * idf(target_node)

High-degree hubs (e.g. 头痛) get low IDF → edges pointing to them are
down-weighted, so rare discriminating symptoms dominate PPR diffusion.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from .base import BaseTool


class SimpleGraph:
    """Lightweight undirected graph for PPR."""

    def __init__(self):
        self.adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.nodes: Set[str] = set()

    def add_edge(self, src: str, dst: str, weight: float):
        self.adj[src].append((dst, weight))
        self.adj[dst].append((src, weight))
        self.nodes.add(src)
        self.nodes.add(dst)

    def neighbors(self, node: str) -> List[Tuple[str, float]]:
        return self.adj.get(node, [])


def idf(N: int, degree: int) -> float:
    """idf(v) = log(1 + N / (1 + degree(v)))"""
    return math.log(1 + N / (1 + degree))


def personalized_pagerank(
    graph: SimpleGraph,
    seed_nodes: List[str],
    alpha: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """PPR power iteration with personalization on seed nodes."""
    nodes = list(graph.nodes)
    if not nodes:
        return {}

    seed_in = [s for s in seed_nodes if s in graph.nodes]
    if not seed_in:
        return {n: 0.0 for n in nodes}

    mass = 1.0 / len(seed_in)
    personal = {n: (mass if n in set(seed_in) else 0.0) for n in nodes}

    r = {n: 1.0 / len(nodes) for n in nodes}
    for _ in range(max_iter):
        new_r = {n: (1 - alpha) * personal.get(n, 0.0) for n in nodes}
        for src in nodes:
            out = graph.neighbors(src)
            total_w = sum(w for _, w in out)
            if total_w == 0:
                continue
            for dst, w in out:
                new_r[dst] += alpha * r[src] * (w / total_w)

        # Redistribute dangling mass equally
        mass_diff = 1.0 - sum(new_r.values())
        if abs(mass_diff) > 1e-10:
            for n in nodes:
                new_r[n] += mass_diff / len(nodes)

        if all(abs(new_r[n] - r[n]) < tol for n in nodes):
            return new_r
        r = new_r
    return r


class PPRReasonerTool(BaseTool):
    name = "ppr_reasoner"
    description = "PPR 多跳推理检索。从已知症状/疾病出发，沿知识图谱做 2-hop 扩散，用个性化 PageRank 发现与种子实体有强关联的潜在疾病。适合鉴别诊断、从多个零散症状推导可能病因。IDF 加权压制高频常见症状的噪声，让罕见鉴别性症状主导推理方向。可按实体类型过滤结果（如只返回疾病类）。输入: {seed_entities: List[str], top_k: int, filter_types: Optional[List[str]]}。返回: 按相关性排序的关联实体列表（含 PPR 分数和实体类型）。"

    def __init__(self, kg_backend, alpha=0.5, use_idf=True):
        self.kg = kg_backend
        self.alpha = alpha
        self.use_idf = use_idf

    async def _execute(self, input_data: Any) -> Dict[str, Any]:
        seeds = input_data.get("seed_entities", [])
        top_k = input_data.get("top_k", 10)
        filter_types = input_data.get("filter_types", None)

        graph, node_types = await self._build_graph(seeds)
        ranks = personalized_pagerank(graph, seeds, alpha=self.alpha)

        results = sorted(ranks.items(), key=lambda x: -x[1])
        seed_set = set(seeds)
        results = [(n, s) for n, s in results if n not in seed_set]
        if filter_types:
            results = [(n, s) for n, s in results if node_types.get(n) in filter_types]

        top = results[:top_k]
        relevant_concepts = [{"entity": n, "ppr_score": round(s, 6)} for n, s in top]

        return {
            "relevant_concepts": relevant_concepts,
            "node_types": node_types,
            "use_idf": self.use_idf,
            "seed_entities": seeds,
            "alpha": self.alpha,
        }

    async def _build_graph(self, seeds: List[str]) -> Tuple[SimpleGraph, Dict[str, str]]:
        """BFS 2-hop expansion with IDF weighting on edge weights."""
        node_types: Dict[str, str] = {}
        node_degrees: Dict[str, int] = defaultdict(int)
        raw_edges: List[Tuple[str, str, float]] = []
        visited: Set[str] = set()
        frontier = list(seeds)

        for hop in range(2):
            next_frontier = []
            for entity in frontier:
                if entity in visited:
                    continue
                visited.add(entity)
                neighbors = await self.kg.query_neighbors(entity)
                for nb in neighbors:
                    target = nb["target"]
                    base_w = nb.get("weight", 0.5)
                    target_type = nb.get("target_type", "unknown")
                    node_types[target] = target_type
                    node_degrees[target] += 1
                    raw_edges.append((entity, target, base_w))
                    if target not in visited:
                        next_frontier.append(target)
            frontier = next_frontier

        # Collect all nodes for N computation
        all_nodes = set(visited)
        for src, dst, _ in raw_edges:
            all_nodes.add(src)
            all_nodes.add(dst)
        N = len(all_nodes)

        # Build graph with IDF-weighted edges
        G = SimpleGraph()
        for src, dst, base_w in raw_edges:
            w = base_w * idf(N, node_degrees[dst]) if self.use_idf else base_w
            G.add_edge(src, dst, w)

        return G, node_types