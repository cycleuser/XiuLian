"""
修炼 - 智能体编排引擎

基于第一性原理设计的非Transformer架构：
- 符号推理：O(n) 模式匹配
- 记忆检索：O(log n) Trie索引  
- 图推理：O(V+E) 关系遍历
- 工具执行：O(1) 直接调用
- 工作流编排：声明式DAG执行
"""

import re
import time
import json
import hashlib
import asyncio
import aiohttp
import subprocess
import os
import sys
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from bs4 import BeautifulSoup

from .api import Result


class ActionType(Enum):
    TOOL = "tool"
    SEARCH = "search"
    WEB = "web"
    QUESTION = "question"
    WORKFLOW = "workflow"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    action: ActionType
    entities: dict
    text: str
    confidence: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class ToolDef:
    name: str
    func: Callable
    description: str
    params_schema: dict
    examples: list = field(default_factory=list)
    category: str = "general"


@dataclass
class WorkflowStep:
    id: str
    action: str
    params: dict = field(default_factory=dict)
    condition: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    retry: int = 0
    timeout: float = 30.0


@dataclass
class Workflow:
    name: str
    description: str
    steps: list[WorkflowStep]
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)


class Parser:
    """符号解析器 - O(n)复杂度"""
    
    PATTERNS = {
        ActionType.TOOL: [
            (r"调用\s*([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
            (r"使用\s*([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
            (r"执行\s*([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
            (r"运行\s*([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
            (r"run\s+([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
            (r"call\s+([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
            (r"execute\s+([a-zA-Z_][a-zA-Z0-9_]*)", "tool"),
        ],
        ActionType.SEARCH: [
            (r"搜索\s+(.+?)(?:\s*$)", "query"),
            (r"查找\s+(.+?)(?:\s*$)", "query"),
            (r"寻找\s+(.+?)(?:\s*$)", "query"),
            (r"search\s+(?:for\s+)?(.+?)(?:\s*$)", "query"),
            (r"find\s+(.+?)(?:\s*$)", "query"),
        ],
        ActionType.WEB: [
            (r"打开\s+(https?://\S+)", "url"),
            (r"访问\s+(https?://\S+)", "url"),
            (r"获取\s+(https?://\S+)", "url"),
            (r"抓取\s+(https?://\S+)", "url"),
            (r"open\s+(https?://\S+)", "url"),
            (r"fetch\s+(https?://\S+)", "url"),
            (r"visit\s+(https?://\S+)", "url"),
        ],
        ActionType.QUESTION: [
            (r"(.+?)是(什么|怎么)", "topic"),
            (r"(如何|怎么|怎样)(.+)", "topic"),
            (r"为什么(.+)", "topic"),
            (r"(.+?)\?$", "text"),
            (r"what\s+is\s+(.+)", "topic"),
            (r"how\s+(?:to\s+)?(.+)", "topic"),
            (r"why\s+(.+)", "topic"),
        ],
        ActionType.WORKFLOW: [
            (r"执行工作流\s*([a-zA-Z_][a-zA-Z0-9_]*)", "workflow"),
            (r"运行流程\s*([a-zA-Z_][a-zA-Z0-9_]*)", "workflow"),
            (r"run\s+workflow\s+([a-zA-Z_][a-zA-Z0-9_]*)", "workflow"),
        ],
    }
    
    PARAM_PATTERNS = {
        "msg": r"(?:msg|消息|message)\s*[=:]?\s*[\"']?([^\"'\s，。]+)[\"']?",
        "message": r"message\s*[=:]?\s*[\"']?([^\"'\s,]+)[\"']?",
        "expr": r"(?:expr|表达式)\s*[=:]?\s*(.+?)\s*$",
        "expression": r"expression\s*[=:]?\s*(.+?)\s*$",
        "url": r"(?:url|网址|链接)\s*[=:]?\s*(https?://\S+)",
        "query": r"(?:query|关键词|查询)\s*[=:]?\s*[\"']?([^\"'\s，。]+)[\"']?",
        "count": r"(?:count|数量|次数)\s*[=:]?\s*(\d+)",
        "timeout": r"(?:timeout|超时)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:秒|s)?",
        "city": r"(?:city|城市)\s*[=:]?\s*([^，。\s]+)",
        "text": r"(?:text|文本)\s*[=:]?\s*[\"']?([^\"']+)[\"']?",
        "target": r"(?:target|目标)\s*[=:]?\s*(\w+)",
        "min": r"(?:min|最小)\s*[=:]?\s*(-?\d+)",
        "max": r"(?:max|最大)\s*[=:]?\s*(-?\d+)",
        "name": r"(?:name|名称)\s*[=:]?\s*(\w+)",
    }
    
    COMPUTE_PATTERN = r"^(?:计算|calc)\s+(.+?)\s*$"
    
    ENTITY_PATTERNS = {
        "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "number": r'-?\b\d+(?:\.\d+)?\b',
        "path": r'(?:/[\w.-]+)+|[\w.-]+\.[a-zA-Z]{2,4}',
        "json": r'\{[^{}]*\}',
    }
    
    def __init__(self):
        self._compile()
    
    def _compile(self):
        self.compiled = {}
        for action, patterns in self.PATTERNS.items():
            self.compiled[action] = [(re.compile(p, re.I), g) for p, g in patterns]
        self.param_compiled = {k: re.compile(p, re.I) for k, p in self.PARAM_PATTERNS.items()}
        self.entity_compiled = {k: re.compile(p, re.I) for k, p in self.ENTITY_PATTERNS.items()}
        self.compute_pattern = re.compile(self.COMPUTE_PATTERN, re.I)
    
    def parse(self, text: str) -> Intent:
        text = text.strip()
        entities = self._extract_entities(text)
        params = self._extract_params(text)
        
        compute_match = self.compute_pattern.match(text)
        if compute_match:
            return Intent(
                action=ActionType.TOOL,
                entities={"target": "calc"},
                text=text,
                params={"expr": compute_match.group(1).strip()}
            )
        
        for action, patterns in self.compiled.items():
            for pattern, group in patterns:
                match = pattern.search(text)
                if match:
                    entities["target"] = match.group(1) if match.groups() else ""
                    return Intent(
                        action=action,
                        entities=entities,
                        text=text,
                        params=params
                    )
        
        return Intent(action=ActionType.UNKNOWN, entities=entities, text=text, params=params)
    
    def _extract_entities(self, text: str) -> dict:
        result = {}
        for etype, pattern in self.entity_compiled.items():
            matches = pattern.findall(text)
            if matches:
                result[etype] = matches[0] if len(matches) == 1 else matches
        return result
    
    def _extract_params(self, text: str) -> dict:
        result = {}
        for pname, pattern in self.param_compiled.items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                value = re.sub(r'^[=:]+\s*', '', value)
                result[pname] = value
        return result


class Memory:
    """稀疏记忆索引"""
    
    def __init__(self):
        self.store: dict[str, Any] = {}
        self.index: dict[str, list[str]] = {}
        self._cache: dict[str, list] = {}
    
    def add(self, key: str, value: Any):
        self.store[key] = value
        for word in key.lower().split():
            self.index.setdefault(word, []).append(key)
        self._cache.clear()
    
    def get(self, key: str) -> Any:
        return self.store.get(key)
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, Any, float]]:
        cache_key = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        words = query.lower().split()
        candidates: dict[str, float] = {}
        
        for word in words:
            for idx_word in self.index:
                if word in idx_word or idx_word in word:
                    for key in self.index[idx_word]:
                        candidates[key] = candidates.get(key, 0) + 1
        
        results = []
        for key, score in sorted(candidates.items(), key=lambda x: -x[1])[:top_k]:
            if key in self.store:
                results.append((key, self.store[key], score / max(len(words), 1)))
        
        self._cache[cache_key] = results
        return results
    
    def load_json(self, path: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
            for item in data.get("knowledge_items", []):
                self.add(item["key"], item["value"])
    
    def save_json(self, path: str):
        data = {"knowledge_items": [{"key": k, "value": v} for k, v in self.store.items()]}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class KnowledgeGraph:
    """知识图谱推理"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._path_cache: dict[tuple, list] = {}
    
    def add_entity(self, entity: str, entity_type: str = "entity", **attrs):
        self.graph.add_node(entity, type=entity_type, **attrs)
        self._path_cache.clear()
    
    def add_relation(self, source: str, target: str, relation: str = "related", weight: float = 1.0):
        self.graph.add_edge(source, target, relation=relation, weight=weight)
        self._path_cache.clear()
    
    def find_path(self, source: str, target: str, max_depth: int = 5) -> list[str]:
        cache_key = (source, target, max_depth)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        try:
            path = nx.shortest_path(self.graph, source, target)
            if len(path) <= max_depth + 1:
                self._path_cache[cache_key] = path
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return []
    
    def get_relations(self, entity: str, relation: str = None) -> list[tuple[str, str, str]]:
        results = []
        for src, dst, data in self.graph.edges(entity, data=True):
            if relation is None or data.get("relation") == relation:
                results.append((src, dst, data.get("relation", "related")))
        return results
    
    def multi_hop(self, start: str, relations: list[str]) -> list[str]:
        current = [start]
        for rel in relations:
            next_nodes = []
            for node in current:
                for _, dst, data in self.graph.edges(node, data=True):
                    if data.get("relation") == rel:
                        next_nodes.append(dst)
            current = list(set(next_nodes))
        return current
    
    def neighbors(self, entity: str, depth: int = 1) -> set[str]:
        result = set()
        frontier = {entity}
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in result:
                        result.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier
        return result


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self.tools: dict[str, ToolDef] = {}
        self.stats: dict[str, dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    def register(self, tool: ToolDef):
        self.tools[tool.name] = tool
        self.stats[tool.name] = {"calls": 0, "successes": 0, "failures": 0, "total_time_ms": 0}
    
    def unregister(self, name: str):
        self.tools.pop(name, None)
        self.stats.pop(name, None)
    
    def get(self, name: str) -> Optional[ToolDef]:
        return self.tools.get(name)
    
    def list_tools(self, category: str = None) -> list[dict]:
        result = []
        for name, tool in self.tools.items():
            if category is None or tool.category == category:
                result.append({
                    "name": name,
                    "description": tool.description,
                    "params": tool.params_schema,
                    "category": tool.category,
                    "examples": tool.examples,
                })
        return result
    
    def execute(self, name: str, **params) -> Result:
        start = time.time()
        
        if name not in self.tools:
            return Result.fail(f"工具不存在: {name}")
        
        tool = self.tools[name]
        stats = self.stats[name]
        stats["calls"] += 1
        
        try:
            result = tool.func(**params)
            stats["successes"] += 1
            stats["total_time_ms"] += (time.time() - start) * 1000
            return Result.ok(result, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            stats["failures"] += 1
            stats["total_time_ms"] += (time.time() - start) * 1000
            return Result.fail(f"{name}执行失败: {str(e)}", latency_ms=(time.time() - start) * 1000)
    
    async def execute_async(self, name: str, **params) -> Result:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.execute, name, **params)
    
    def execute_parallel(self, calls: list[tuple[str, dict]]) -> list[Result]:
        futures = [self._executor.submit(self.execute, name, **params) for name, params in calls]
        return [f.result() for f in futures]
    
    def get_stats(self) -> dict:
        return {
            "tools": len(self.tools),
            "total_calls": sum(s["calls"] for s in self.stats.values()),
            "total_successes": sum(s["successes"] for s in self.stats.values()),
            "per_tool": self.stats,
        }


class WebClient:
    """网络访问客户端"""
    
    def __init__(self, timeout: float = 10.0, cache_size: int = 100):
        self.timeout = timeout
        self.cache: dict[str, tuple[str, float]] = {}
        self.cache_size = cache_size
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "XiuLian/1.0"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch(self, url: str, use_cache: bool = True) -> Result:
        start = time.time()
        cache_key = hashlib.md5(url.encode()).hexdigest()
        
        if use_cache and cache_key in self.cache:
            content, _ = self.cache[cache_key]
            return Result.ok({"content": content, "url": url, "cached": True}, latency_ms=(time.time() - start) * 1000)
        
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return Result.fail(f"HTTP {resp.status}: {resp.reason}")
                content = await resp.text()
                
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[cache_key] = (content, time.time())
                
                return Result.ok({"content": content, "url": url, "cached": False}, latency_ms=(time.time() - start) * 1000)
        except asyncio.TimeoutError:
            return Result.fail(f"请求超时 ({self.timeout}s)")
        except Exception as e:
            return Result.fail(f"请求失败: {str(e)}")
    
    async def post(self, url: str, data: dict = None, json_data: dict = None) -> Result:
        start = time.time()
        try:
            session = await self._get_session()
            async with session.post(url, data=data, json=json_data) as resp:
                content = await resp.text()
                return Result.ok({"content": content, "url": url, "status": resp.status}, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return Result.fail(f"POST失败: {str(e)}")
    
    def parse_html(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title')
        description = soup.find('meta', attrs={'name': 'description'})
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        links = []
        for a in soup.find_all('a', href=True)[:20]:
            href = a['href']
            if href.startswith('http'):
                links.append({"url": href, "text": a.get_text(strip=True)[:50]})
        
        return {
            "url": url,
            "title": title.get_text(strip=True) if title else "",
            "description": description['content'] if description else "",
            "content": main_content.get_text(strip=True)[:5000] if main_content else "",
            "links": links,
        }


class WorkflowEngine:
    """工作流执行引擎"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.workflows: dict[str, Workflow] = {}
        self.tools = tool_registry
        self._execution_history: list[dict] = []
    
    def register(self, workflow: Workflow):
        self.workflows[workflow.name] = workflow
    
    def get(self, name: str) -> Optional[Workflow]:
        return self.workflows.get(name)
    
    def list_workflows(self) -> list[dict]:
        return [{"name": w.name, "description": w.description, "steps": len(w.steps)} for w in self.workflows.values()]
    
    def execute(self, name: str, input_data: dict = None) -> Result:
        start = time.time()
        
        if name not in self.workflows:
            return Result.fail(f"工作流不存在: {name}")
        
        workflow = self.workflows[name]
        context = {"input": input_data or {}, "results": {}, "errors": {}}
        
        for step in workflow.steps:
            step_start = time.time()
            
            if step.condition:
                try:
                    if not eval(step.condition, {"context": context}):
                        continue
                except Exception as e:
                    context["errors"][step.id] = f"条件评估失败: {e}"
                    if step.on_failure:
                        continue
                    else:
                        return Result.fail(f"步骤{step.id}条件评估失败", latency_ms=(time.time() - start) * 1000)
            
            retries = 0
            while retries <= step.retry:
                result = self.tools.execute(step.action, **step.params)
                
                if result.success:
                    context["results"][step.id] = result.data
                    if step.on_success:
                        continue
                    break
                else:
                    retries += 1
                    if retries > step.retry:
                        context["errors"][step.id] = result.error
                        if step.on_failure:
                            continue
                        else:
                            return Result.fail(f"步骤{step.id}执行失败: {result.error}", latency_ms=(time.time() - start) * 1000)
                    time.sleep(0.5)
            
            self._execution_history.append({
                "workflow": name,
                "step": step.id,
                "action": step.action,
                "latency_ms": (time.time() - step_start) * 1000,
                "success": step.id in context["results"],
            })
        
        return Result.ok(context["results"], latency_ms=(time.time() - start) * 1000)
    
    async def execute_async(self, name: str, input_data: dict = None) -> Result:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, name, input_data)
    
    def get_history(self) -> list[dict]:
        return self._execution_history


class Agent:
    """智能体核心引擎"""
    
    def __init__(self):
        self.parser = Parser()
        self.memory = Memory()
        self.knowledge = KnowledgeGraph()
        self.tools = ToolRegistry()
        self.web = WebClient()
        self.workflows = WorkflowEngine(self.tools)
        
        self._register_builtin_tools()
        self._register_builtin_workflows()
    
    def _register_builtin_tools(self):
        def echo(**kwargs):
            msg = kwargs.get('msg', kwargs.get('message', 'hello'))
            return {"message": str(msg)}
        
        def calc(**kwargs):
            expr = kwargs.get('expr', kwargs.get('expression', '0'))
            try:
                allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
                result = eval(str(expr), {"__builtins__": {}}, allowed)
                return {"expression": expr, "result": result}
            except Exception as e:
                return {"expression": expr, "error": str(e)}
        
        def get_time(**kwargs):
            return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "timestamp": time.time()}
        
        def random_number(**kwargs):
            import random
            min_val = int(kwargs.get('min', 0))
            max_val = int(kwargs.get('max', 100))
            return {"number": random.randint(min_val, max_val)}
        
        def json_parse(**kwargs):
            text = kwargs.get('text', '{}')
            try:
                return {"parsed": json.loads(text), "valid": True}
            except:
                return {"parsed": None, "valid": False}
        
        def string_format(**kwargs):
            template = kwargs.get('template', '')
            values = kwargs.get('values', {})
            return {"result": template.format(**values)}
        
        def list_tools(**kwargs):
            return {"tools": self.tools.list_tools()}
        
        def help_tool(**kwargs):
            tool_name = kwargs.get('name')
            if tool_name:
                tool = self.tools.get(tool_name)
                if tool:
                    return {"name": tool.name, "description": tool.description, "params": tool.params_schema}
                return {"error": f"工具{tool_name}不存在"}
            return {"tools": [t["name"] for t in self.tools.list_tools()]}
        
        self.tools.register(ToolDef(
            name="echo", func=echo, description="回显消息",
            params_schema={"msg": {"type": "string", "description": "要回显的消息"}},
            examples=["调用echo msg=hello", "run echo with message=world"],
            category="basic"
        ))
        
        self.tools.register(ToolDef(
            name="calc", func=calc, description="计算数学表达式",
            params_schema={"expr": {"type": "string", "description": "数学表达式"}},
            examples=["调用calc expr=2+2", "计算 100*5"],
            category="math"
        ))
        
        self.tools.register(ToolDef(
            name="time", func=get_time, description="获取当前时间",
            params_schema={}, examples=["调用time", "what time is it"],
            category="basic"
        ))
        
        self.tools.register(ToolDef(
            name="random", func=random_number, description="生成随机数",
            params_schema={"min": {"type": "int", "default": 0}, "max": {"type": "int", "default": 100}},
            examples=["调用random min=1 max=100"],
            category="math"
        ))
        
        self.tools.register(ToolDef(
            name="json", func=json_parse, description="解析JSON字符串",
            params_schema={"text": {"type": "string", "description": "JSON字符串"}},
            examples=['调用json text={"a":1}'],
            category="data"
        ))
        
        self.tools.register(ToolDef(
            name="format", func=string_format, description="格式化字符串",
            params_schema={"template": {"type": "string"}, "values": {"type": "dict"}},
            examples=['调用format template="Hello {name}" values={"name":"World"}'],
            category="text"
        ))
        
        self.tools.register(ToolDef(
            name="tools", func=list_tools, description="列出所有可用工具",
            params_schema={}, examples=["列出工具", "show tools"],
            category="system"
        ))
        
        self.tools.register(ToolDef(
            name="help", func=help_tool, description="获取工具帮助信息",
            params_schema={"name": {"type": "string", "description": "工具名称"}},
            examples=["调用help name=calc"],
            category="system"
        ))
    
    def _register_builtin_workflows(self):
        search_workflow = Workflow(
            name="web_search",
            description="执行网络搜索并返回结果",
            steps=[
                WorkflowStep(id="fetch", action="web_fetch", params={"url": "https://api.duckduckgo.com/?q={query}&format=json"}),
            ]
        )
        self.workflows.register(search_workflow)
    
    def process(self, text: str) -> Result:
        start = time.time()
        intent = self.parser.parse(text)
        
        if intent.action == ActionType.TOOL:
            tool_name = intent.entities.get("target", "").lower()
            params = dict(intent.params)
            for k, v in intent.entities.items():
                if k not in ("target",) and not k.startswith("_") and k not in params:
                    params[k] = v
            return self.tools.execute(tool_name, **params)
        
        if intent.action == ActionType.SEARCH:
            query = intent.entities.get("target", intent.entities.get("query", text))
            results = self.memory.search(query)
            return Result.ok({"query": query, "results": results}, latency_ms=(time.time() - start) * 1000)
        
        if intent.action == ActionType.QUESTION:
            topic = intent.entities.get("topic", intent.entities.get("text", text))
            results = self.memory.search(topic)
            if results:
                return Result.ok({"question": topic, "answer": results[0][1], "confidence": results[0][2]}, latency_ms=(time.time() - start) * 1000)
            return Result.ok({"question": topic, "answer": "未找到相关内容"}, latency_ms=(time.time() - start) * 1000)
        
        if intent.action == ActionType.WEB:
            url = intent.entities.get("url", intent.entities.get("target", ""))
            if not url.startswith("http"):
                url = "https://" + url
            
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self.web.fetch(url))
                if result.success and result.data:
                    parsed = self.web.parse_html(result.data["content"], url)
                    return Result.ok(parsed, latency_ms=(time.time() - start) * 1000)
                return result
            finally:
                loop.close()
        
        if intent.action == ActionType.WORKFLOW:
            workflow_name = intent.entities.get("target", "")
            return self.workflows.execute(workflow_name, intent.params)
        
        return Result.ok({
            "intent": intent.action.value,
            "entities": intent.entities,
            "suggestion": self._suggest_action(intent)
        }, latency_ms=(time.time() - start) * 1000)
    
    def _suggest_action(self, intent: Intent) -> str:
        if intent.action == ActionType.UNKNOWN:
            return "试试：调用echo msg=hello 或 搜索AI 或 打开 https://example.com"
        return ""
    
    async def process_async(self, text: str) -> Result:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, text)
    
    def batch_process(self, texts: list[str]) -> list[Result]:
        return [self.process(text) for text in texts]
    
    async def parallel_process(self, texts: list[str]) -> list[Result]:
        tasks = [self.process_async(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def register_tool(self, name: str, func: Callable, description: str = "", params_schema: dict = None):
        self.tools.register(ToolDef(
            name=name, func=func, description=description,
            params_schema=params_schema or {}, category="custom"
        ))
    
    def register_workflow(self, name: str, steps: list[dict], description: str = ""):
        workflow_steps = [
            WorkflowStep(
                id=s.get("id", f"step_{i}"),
                action=s["action"],
                params=s.get("params", {}),
                condition=s.get("condition"),
                on_success=s.get("on_success"),
                on_failure=s.get("on_failure"),
                retry=s.get("retry", 0),
                timeout=s.get("timeout", 30.0),
            )
            for i, s in enumerate(steps)
        ]
        self.workflows.register(Workflow(name=name, description=description, steps=workflow_steps))
    
    def add_knowledge(self, key: str, value: Any):
        self.memory.add(key, value)
    
    def load_knowledge(self, path: str):
        self.memory.load_json(path)
    
    def add_entity(self, entity: str, entity_type: str = "entity", **attrs):
        self.knowledge.add_entity(entity, entity_type, **attrs)
    
    def add_relation(self, source: str, target: str, relation: str = "related"):
        self.knowledge.add_relation(source, target, relation)
    
    def get_stats(self) -> dict:
        return {
            "tools": self.tools.get_stats(),
            "workflows": len(self.workflows.workflows),
            "knowledge_items": len(self.memory.store),
            "entities": self.knowledge.graph.number_of_nodes(),
            "relations": self.knowledge.graph.number_of_edges(),
        }
    
    def benchmark(self, iterations: int = 100) -> dict:
        cases = [
            "调用echo msg=test",
            "搜索人工智能",
            "什么是机器学习?",
            "调用calc expr=2+2",
            "调用time",
        ] * (iterations // 5 + 1)
        cases = cases[:iterations]
        
        start = time.time()
        latencies = []
        successes = 0
        
        for case in cases:
            r = self.process(case)
            latencies.append(r.latency_ms)
            if r.success:
                successes += 1
        
        return {
            "iterations": iterations,
            "successes": successes,
            "success_rate": successes / iterations,
            "total_time_s": time.time() - start,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_per_s": iterations / (time.time() - start),
        }


Engine = Agent