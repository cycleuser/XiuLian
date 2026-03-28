"""
修炼 - 轻量级符号推理引擎

非Transformer架构，专注于工具调用和网络访问。
复杂度 O(n log n)，内存 <2GB，延迟 <10ms。
"""

__version__ = "1.0.0"

from .core import Engine, Intent, Memory, Graph, Tool, Web
from .api import Result

__all__ = ["Engine", "Intent", "Memory", "Graph", "Tool", "Web", "Result"]