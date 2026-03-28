"""统一结果类型"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Result:
    """统一的操作结果类型
    
    Attributes:
        success: 操作是否成功
        data: 成功时返回的数据
        error: 失败时的错误信息
        latency_ms: 操作耗时（毫秒）
        metadata: 额外的元数据
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any = None, latency_ms: float = 0.0, metadata: dict = None) -> "Result":
        return cls(success=True, data=data, latency_ms=latency_ms, metadata=metadata or {})
    
    @classmethod
    def fail(cls, error: str, latency_ms: float = 0.0, metadata: dict = None) -> "Result":
        return cls(success=False, error=error, latency_ms=latency_ms, metadata=metadata or {})
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }
    
    def __bool__(self) -> bool:
        return self.success
    
    def __repr__(self) -> str:
        if self.success:
            return f"Result(ok, {self.latency_ms:.2f}ms)"
        return f"Result(fail: {self.error})"