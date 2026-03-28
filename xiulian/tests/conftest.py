"""测试配置"""
import pytest
from xiulian.core import Engine, Memory, Graph

@pytest.fixture
def engine():
    return Engine()

@pytest.fixture
def memory():
    m = Memory()
    m.add("AI", {"def": "人工智能"})
    m.add("ML", {"def": "机器学习"})
    return m

@pytest.fixture
def graph():
    g = Graph()
    g.add("AI"); g.add("ML"); g.add("DL")
    g.link("AI", "ML"); g.link("ML", "DL")
    return g