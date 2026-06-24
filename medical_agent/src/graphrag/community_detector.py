"""
graphrag/community_detector.py — 社区检测

生产环境用 Leiden（igraph + leidenalg），见 scripts/build_graphrag_index.py。
本模块提供一个**纯 Python、无外部依赖、确定性**的轻量实现，用于:
- 在 mock KG 上演示"社区检测 -> 分层"的完整流程
- 单元测试（不必安装 igraph）

算法: 加权标签传播 (Label Propagation)。
- 每个节点初始化为独立标签
- 迭代中，每个节点采用其邻居中加权票数最高的标签
- 用"分辨率"参数控制合并倾向，模拟 Leiden 的多分辨率分层

注意: 这是教学/演示级实现，不追求与 Leiden 的 modularity 对齐。
真实数据请用 scripts/build_graphrag_index.py 的 Leiden 版本。

实现状态: ✅ 完整（mock/演示级）
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


class CommunityDetector:
    """
    加权标签传播社区检测，支持多分辨率分层。

    用法:
        edges = [("头痛","脑膜炎",0.3), ("发烧","脑膜炎",0.2), ...]
        detector = CommunityDetector(edges)
        levels = detector.detect_hierarchical(resolutions=[0.5, 1.0, 1.5])
        # levels = {"l0": {node: comm_id}, "l1": {...}, "l2": {...}}
    """

    def __init__(self, edges: List[Tuple[str, str, float]]):
        # 无向加权邻接表
        self.adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.nodes: List[str] = []
        seen = set()
        for src, dst, w in edges:
            self.adj[src][dst] += w
            self.adj[dst][src] += w
            for n in (src, dst):
                if n not in seen:
                    seen.add(n)
                    self.nodes.append(n)

    def detect(self, resolution: float = 1.0, max_iter: int = 50) -> Dict[str, int]:
        """
        单层社区检测，返回 {node: community_id}。

        resolution 越大 -> 越倾向于细碎社区（更难合并）。
        实现上用 resolution 缩放"采纳邻居标签"的阈值。
        """
        # 确定性初始化：按节点出现顺序赋予唯一标签
        label: Dict[str, int] = {n: i for i, n in enumerate(self.nodes)}

        for _ in range(max_iter):
            changed = False
            # 确定性遍历顺序（按节点固定顺序，保证可复现）
            for node in self.nodes:
                neighbors = self.adj.get(node, {})
                if not neighbors:
                    continue
                # 统计邻居各标签的加权票数
                votes: Dict[int, float] = defaultdict(float)
                for nb, w in neighbors.items():
                    votes[label[nb]] += w
                # resolution 缩放：自身标签获得一个"惯性"加成，
                # resolution 越大惯性越大 -> 越不容易被邻居合并 -> 社区更细
                votes[label[node]] += resolution * 0.5

                # 选票数最高的标签（平票时取 id 最小，保证确定性）
                best_label = min(
                    (lbl for lbl, v in votes.items() if v == max(votes.values())),
                )
                if best_label != label[node]:
                    label[node] = best_label
                    changed = True
            if not changed:
                break

        # 重新编号社区 id，使其连续（0,1,2,...）
        return self._renumber(label)

    def detect_hierarchical(
        self,
        resolutions: List[float] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        多分辨率分层社区检测。

        Args:
            resolutions: 分辨率列表，从小到大对应 粗->细。
                         默认 [0.5, 1.0, 1.5] 对应 l0(粗)/l1(中)/l2(细)。

        Returns:
            {"l0": {node: comm}, "l1": {...}, "l2": {...}}
        """
        if resolutions is None:
            resolutions = [0.5, 1.0, 1.5]
        result = {}
        for level, res in enumerate(resolutions):
            result[f"l{level}"] = self.detect(resolution=res)
        return result

    @staticmethod
    def _renumber(label: Dict[str, int]) -> Dict[str, int]:
        """把任意标签重映射为连续的 0,1,2,... （按首次出现顺序）"""
        remap: Dict[int, int] = {}
        out: Dict[str, int] = {}
        for node, lbl in label.items():
            if lbl not in remap:
                remap[lbl] = len(remap)
            out[node] = remap[lbl]
        return out

    def community_members(self, labels: Dict[str, int]) -> Dict[int, List[str]]:
        """把 {node: comm} 反转为 {comm: [nodes]}"""
        members: Dict[int, List[str]] = defaultdict(list)
        for node, comm in labels.items():
            members[comm].append(node)
        return dict(members)

    def num_communities(self, labels: Dict[str, int]) -> int:
        return len(set(labels.values()))
