"""修炼测试"""

import pytest
import time
from xiulian.core import Engine, SymbolicParser, Memory, Graph, Tool
from xiulian.api import Result

class TestParser:
    def test_tool_intent(self):
        p = SymbolicParser()
        r = p.parse("调用echo")
        assert r.action == "tool"
    
    def test_search_intent(self):
        p = SymbolicParser()
        r = p.parse("搜索AI")
        assert r.action == "search"
    
    def test_web_intent(self):
        p = SymbolicParser()
        r = p.parse("打开 https://example.com")
        assert r.action == "web"
    
    def test_question_intent(self):
        p = SymbolicParser()
        r = p.parse("什么是AI?")
        assert r.action == "question"
    
    def test_entity_url(self):
        p = SymbolicParser()
        r = p.parse("访问 https://test.com")
        assert "url" in r.entities
    
    def test_speed(self):
        p = SymbolicParser()
        start = time.time()
        for _ in range(1000):
            p.parse("这是一个测试")
        assert time.time() - start < 1.0

class TestMemory:
    def test_add_search(self):
        m = Memory()
        m.add("python", {"type": "语言"})
        r = m.search("python")
        assert len(r) > 0
    
    def test_relevance(self):
        m = Memory()
        m.add("machine learning", {"def": "ML"})
        m.add("deep learning", {"def": "DL"})
        r = m.search("learning")
        assert len(r) == 2

class TestGraph:
    def test_add_path(self):
        g = Graph()
        g.add("A")
        g.add("B")
        g.link("A", "B")
        assert g.path("A", "B") == ["A", "B"]
    
    def test_neighbors(self):
        g = Graph()
        g.add("A"); g.add("B"); g.add("C")
        g.link("A", "B"); g.link("B", "C")
        assert "B" in g.neighbors("A")

class TestTool:
    def test_register_execute(self):
        t = Tool()
        t.register("test", lambda x: x * 2, "测试")
        r = t.execute("test", x=5)
        assert r.success and r.data == 10
    
    def test_unknown_tool(self):
        t = Tool()
        r = t.execute("unknown")
        assert not r.success

class TestEngine:
    def test_process_tool(self):
        e = Engine()
        r = e.process("调用echo msg=hello")
        assert r.success
    
    def test_process_question(self):
        e = Engine()
        e.memory.add("AI", {"def": "人工智能"})
        r = e.process("什么是AI?")
        assert r.success
    
    def test_benchmark(self):
        e = Engine()
        r = e.benchmark(50)
        assert r["avg_latency_ms"] < 50

class TestResult:
    def test_ok(self):
        r = Result.ok({"a": 1})
        assert r.success and r.data == {"a": 1}
    
    def test_fail(self):
        r = Result.fail("错误")
        assert not r.success and r.error == "错误"