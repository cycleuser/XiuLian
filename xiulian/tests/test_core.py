"""修炼测试"""

import pytest
import time
from xiulian.core import Engine, Parser, Memory, KnowledgeGraph, ToolRegistry, ToolDef, ActionType
from xiulian.api import Result

class TestParser:
    def test_tool_intent(self):
        p = Parser()
        r = p.parse("调用echo")
        assert r.action == ActionType.TOOL
    
    def test_search_intent(self):
        p = Parser()
        r = p.parse("搜索 AI")
        assert r.action == ActionType.SEARCH
    
    def test_web_intent(self):
        p = Parser()
        r = p.parse("打开 https://example.com")
        assert r.action == ActionType.WEB
    
    def test_question_intent(self):
        p = Parser()
        r = p.parse("什么是AI?")
        assert r.action == ActionType.QUESTION
    
    def test_entity_url(self):
        p = Parser()
        r = p.parse("访问 https://test.com")
        assert "url" in r.entities
    
    def test_speed(self):
        p = Parser()
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

class TestKnowledgeGraph:
    def test_add_path(self):
        g = KnowledgeGraph()
        g.add_entity("A")
        g.add_entity("B")
        g.add_relation("A", "B")
        assert g.find_path("A", "B") == ["A", "B"]
    
    def test_neighbors(self):
        g = KnowledgeGraph()
        g.add_entity("A"); g.add_entity("B"); g.add_entity("C")
        g.add_relation("A", "B"); g.add_relation("B", "C")
        rels = g.get_relations("A")
        assert len(rels) > 0

class TestToolRegistry:
    def test_register_execute(self):
        t = ToolRegistry()
        tool = ToolDef(
            name="test",
            func=lambda x: x * 2,
            description="测试工具",
            params_schema={"x": "int"}
        )
        t.register(tool)
        r = t.execute("test", x=5)
        assert r.success and r.data == 10
    
    def test_unknown_tool(self):
        t = ToolRegistry()
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