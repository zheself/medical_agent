"""GraphRAG 离线索引模块（mock/演示级实现）。

生产环境的完整流程（Leiden + LLM 摘要 + 向量化 + 写回 Neo4j/Milvus）
见 scripts/build_graphrag_index.py。

本模块提供纯 Python、无外部依赖、确定性的实现，用于演示和单元测试：
- CommunityDetector: 加权标签传播社区检测，支持多分辨率分层
- CommunitySummaryGenerator: 结构化摘要生成 + 回填验证（防幻觉）
"""
from .community_detector import CommunityDetector
from .summary_generator import (
    CommunitySummaryGenerator,
    CommunitySummaryResult,
    infer_theme,
)

__all__ = [
    "CommunityDetector",
    "CommunitySummaryGenerator",
    "CommunitySummaryResult",
    "infer_theme",
]
