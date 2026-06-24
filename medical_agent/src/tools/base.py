"""
tools/base.py — Tool 基类

设计原则：
1. 所有 Tool 都是无状态的纯函数（可并发调用）
2. 统一的 invoke/ainvoke 接口
3. 自动计时和错误捕获
4. 输出统一为 ToolResult

实现状态: ✅ 完整
"""
from __future__ import annotations

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..schemas import ToolResult


class BaseTool(ABC):
    """所有工具的基类"""
    
    name: str = ""
    description: str = ""
    
    @abstractmethod
    async def _execute(self, input_data: Any) -> Any:
        """子类实现具体逻辑，返回原始数据。错误请抛异常。"""
        raise NotImplementedError
    
    async def ainvoke(self, input_data: Any) -> ToolResult:
        """统一的异步调用入口，自动计时和错误处理"""
        start = time.time()
        try:
            data = await self._execute(input_data)
            elapsed_ms = (time.time() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                elapsed_ms=elapsed_ms,
                citations=self._extract_citations(data),
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                elapsed_ms=elapsed_ms,
            )
    
    def invoke(self, input_data: Any) -> ToolResult:
        """同步包装，便于在脚本中使用"""
        return asyncio.run(self.ainvoke(input_data))
    
    def _extract_citations(self, data: Any) -> List[Dict]:
        """子类可重写以提取引用信息（用于 citation accuracy 评测）"""
        return []
    
    def to_description(self) -> Dict[str, str]:
        """供 Planner 看的工具描述"""
        return {"name": self.name, "description": self.description}


class ToolRegistry:
    """工具注册中心，统一管理所有 Tool"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"Tool {tool} 必须设置 name")
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"未注册的 Tool: {name}。已注册: {list(self._tools.keys())}")
        return self._tools[name]
    
    def get_all_descriptions(self) -> List[Dict[str, str]]:
        return [t.to_description() for t in self._tools.values()]
    
    def list_names(self) -> List[str]:
        return list(self._tools.keys())
