"""统一结果类型"""

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Result:
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    
    @classmethod
    def ok(cls, data, latency_ms=0.0):
        return cls(success=True, data=data, latency_ms=latency_ms)
    
    @classmethod
    def fail(cls, error, latency_ms=0.0):
        return cls(success=False, error=error, latency_ms=latency_ms)
    
    def to_dict(self):
        return {"success": self.success, "data": self.data, "error": self.error, "latency_ms": self.latency_ms}