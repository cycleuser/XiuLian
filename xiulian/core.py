"""
修炼核心引擎

基于第一性原理设计：
- 符号推理：O(n) 模式匹配
- 记忆检索：O(log n) Trie索引  
- 图推理：O(V+E) 关系遍历
- 工具执行：O(1) 直接调用
"""

import re
import time
import json
import hashlib
import asyncio
import aiohttp
from typing import Any, Callable
from dataclasses import dataclass, field
from functools import lru_cache

import ahocorasick
import networkx as nx
import numpy as np
from bs4 import BeautifulSoup

from .api import Result


@dataclass
class Intent:
    """解析后的意图"""
    action: str
    entities: dict
    text: str
    confidence: float = 1.0


class SymbolicParser:
    """符号解析器 - O(n) 复杂度"""
    
    PATTERNS = {
        "tool": [
            (r"调用\s*([a-zA-Z_]+)", "call"),
            (r"使用\s*([a-zA-Z_]+)", "use"),
            (r"执行\s*([a-zA-Z_]+)", "exec"),
            (r"run\s+([a-zA-Z_]+)", "call"),
            (r"call\s+([a-zA-Z_]+)", "call"),
            (r"用\s*([a-zA-Z_]+)\s*工具", "use"),
        ],
        "search": [
            (r"搜索\s*(.+)", "query"),
            (r"查找\s*(.+)", "query"),
            (r"search\s+(?:for\s+)?(.+)", "query"),
            (r"find\s+(.+)", "query"),
        ],
        "web": [
            (r"打开\s*(https?://\S+)", "url"),
            (r"访问\s*(https?://\S+)", "url"),
            (r"open\s+(https?://\S+)", "url"),
            (r"fetch\s+(https?://\S+)", "url"),
        ],
        "question": [
            (r"(.+)\?$", "text"),
            (r"什么是\s*(.+)", "topic"),
            (r"如何\s*(.+)", "topic"),
            (r"what\s+is\s+(.+)", "topic"),
            (r"how\s+(?:to\s+)?(.+)", "topic"),
        ],
    }
    
    ENTITY_PATTERNS = {
        "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "number": r'\b\d+(?:\.\d+)?\b',
        "path": r'(?:/[\w.-]+)+|[\w.-]+\.\w{2,4}',
    }
    
    def __init__(self):
        self._compile()
    
    def _compile(self):
        self.compiled = {}
        for action, patterns in self.PATTERNS.items():
            self.compiled[action] = [(re.compile(p, re.I), group) for p, group in patterns]
        self.entities = {k: re.compile(p, re.I) for k, p in self.ENTITY_PATTERNS.items()}
    
    def parse(self, text: str) -> Intent:
        text = text.strip()
        entities = self._extract_entities(text)
        
        for action, patterns in self.compiled.items():
            for pattern, group_name in patterns:
                match = pattern.search(text)
                if match:
                    entities["target"] = match.group(1) if match.groups() else ""
                    return Intent(action=action, entities=entities, text=text)
        
        return Intent(action="unknown", entities=entities, text=text, confidence=0.5)
    
    def _extract_entities(self, text: str) -> dict:
        result = {}
        for etype, pattern in self.entities.items():
            matches = pattern.findall(text)
            if matches:
                result[etype] = matches[0] if len(matches) == 1 else matches
        return result


class Memory:
    """稀疏记忆索引 - O(log n) 检索"""
    
    def __init__(self):
        self.store: dict[str, Any] = {}
        self.index: dict[str, list[str]] = {}
    
    def add(self, key: str, value: Any):
        self.store[key] = value
        for word in key.lower().split():
            if word not in self.index:
                self.index[word] = []
            self.index[word].append(key)
    
    def get(self, key: str) -> Any:
        return self.store.get(key)
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, Any, float]]:
        words = query.lower().split()
        candidates = {}
        
        for word in words:
            if word in self.index:
                for key in self.index[word]:
                    candidates[key] = candidates.get(key, 0) + 1
        
        results = []
        for key, score in sorted(candidates.items(), key=lambda x: -x[1])[:top_k]:
            results.append((key, self.store[key], score / max(len(words), 1)))
        
        return results
    
    def load_json(self, path: str):
        with open(path, encoding='utf-8') as f:
            for item in json.load(f).get("knowledge_items", []):
                self.add(item["key"], item["value"])


class Graph:
    """图推理引擎 - O(V+E) 复杂度"""
    
    def __init__(self):
        self.g = nx.DiGraph()
    
    def add(self, node: str, node_type: str = "entity", **attrs):
        self.g.add_node(node, type=node_type, **attrs)
    
    def link(self, src: str, dst: str, relation: str = "related"):
        self.g.add_edge(src, dst, relation=relation)
    
    def path(self, src: str, dst: str) -> list[str]:
        try:
            return nx.shortest_path(self.g, src, dst)
        except nx.NetworkXNoPath:
            return []
    
    def neighbors(self, node: str, depth: int = 1) -> set[str]:
        result = set()
        frontier = {node}
        for _ in range(depth):
            new_frontier = set()
            for n in frontier:
                for neighbor in self.g.neighbors(n):
                    if neighbor not in result:
                        result.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier
        return result
    
    def query(self, node: str, relation: str) -> list[str]:
        return [dst for _, dst, data in self.g.edges(node, data=True) if data.get("relation") == relation]


class Tool:
    """工具注册与执行"""
    
    def __init__(self):
        self.tools: dict[str, dict] = {}
        self.stats = {"total": 0, "success": 0}
    
    def register(self, name: str, func: Callable, desc: str = "", params: dict = None):
        self.tools[name] = {"func": func, "desc": desc, "params": params or {}}
    
    def execute(self, name: str, **kwargs) -> Result:
        start = time.time()
        self.stats["total"] += 1
        
        if name not in self.tools:
            return Result.fail(f"工具不存在: {name}")
        
        try:
            result = self.tools[name]["func"](**kwargs)
            self.stats["success"] += 1
            return Result.ok(result, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return Result.fail(str(e), latency_ms=(time.time() - start) * 1000)
    
    def list(self) -> list[dict]:
        return [{"name": k, "desc": v["desc"], "params": v["params"]} for k, v in self.tools.items()]


class Web:
    """网络访问模块"""
    
    def __init__(self, timeout: float = 10.0, cache_size: int = 100):
        self.timeout = timeout
        self.cache: dict[str, tuple[str, float]] = {}
        self.cache_size = cache_size
    
    async def fetch(self, url: str, use_cache: bool = True) -> Result:
        start = time.time()
        key = hashlib.md5(url.encode()).hexdigest()
        
        if use_cache and key in self.cache:
            content, _ = self.cache[key]
            return Result.ok({"content": content, "cached": True}, latency_ms=(time.time() - start) * 1000)
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return Result.fail(f"HTTP {resp.status}")
                    content = await resp.text()
                    
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[key] = (content, time.time())
                    
                    return Result.ok({"content": content, "cached": False}, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return Result.fail(str(e))
    
    def parse(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title')
        return {
            "url": url,
            "title": title.get_text(strip=True) if title else "",
            "content": (soup.find('main') or soup.find('body') or soup).get_text(strip=True)[:5000],
        }
    
    async def fetch_and_parse(self, url: str) -> Result:
        result = await self.fetch(url)
        if not result.success:
            return result
        return Result.ok(self.parse(result.data["content"], url), latency_ms=result.latency_ms)


class Engine:
    """修炼主引擎"""
    
    def __init__(self):
        self.parser = SymbolicParser()
        self.memory = Memory()
        self.graph = Graph()
        self.tools = Tool()
        self.web = Web()
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        def echo_tool(**kwargs):
            msg = kwargs.get('msg', kwargs.get('message', 'hello'))
            return {"message": msg}
        
        def calc_tool(**kwargs):
            expr = kwargs.get('expr', '0')
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return {"result": result}
            except:
                return {"error": "Invalid expression"}
        
        def time_tool(**kwargs):
            return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
        
        self.tools.register("echo", echo_tool, "回显消息", {"msg": "string"})
        self.tools.register("calc", calc_tool, "计算表达式", {"expr": "string"})
        self.tools.register("time", time_tool, "获取当前时间")
    
    def process(self, text: str) -> Result:
        start = time.time()
        intent = self.parser.parse(text)
        
        if intent.action == "tool":
            tool_name = intent.entities.get("target", "").lower()
            params = {k: v for k, v in intent.entities.items() if k != "target" and not k.startswith("_")}
            return self.tools.execute(tool_name, **params)
        
        if intent.action == "search":
            results = self.memory.search(intent.entities.get("target", text))
            return Result.ok({"results": results}, latency_ms=(time.time() - start) * 1000)
        
        if intent.action == "question":
            results = self.memory.search(intent.entities.get("topic", text))
            if results:
                return Result.ok({"answer": results[0][1]}, latency_ms=(time.time() - start) * 1000)
            return Result.ok({"answer": "未找到相关内容"}, latency_ms=(time.time() - start) * 1000)
        
        if intent.action == "web":
            url = intent.entities.get("url") or intent.entities.get("target", "")
            if not url.startswith("http"):
                url = "https://" + url
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.web.fetch_and_parse(url))
            finally:
                loop.close()
        
        return Result.ok({"intent": intent.action, "entities": intent.entities}, latency_ms=(time.time() - start) * 1000)
    
    def load_knowledge(self, path: str):
        self.memory.load_json(path)
    
    def benchmark(self, iterations: int = 100) -> dict:
        cases = ["调用echo", "搜索AI", "什么是机器学习?", "打开 https://example.com"] * (iterations // 4 + 1)
        cases = cases[:iterations]
        
        start = time.time()
        latencies = []
        for case in cases:
            r = self.process(case)
            latencies.append(r.latency_ms)
        
        return {
            "iterations": iterations,
            "total_time_s": time.time() - start,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput": iterations / (time.time() - start),
        }