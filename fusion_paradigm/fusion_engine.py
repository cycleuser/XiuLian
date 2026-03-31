"""
融合范式引擎 - Fusion Paradigm Engine
整合传统AI方法(XiuLian)与大模型推理(ollama)

基于第一性原理设计：
- 符号推理：确定性、高效率、可解释
- LLM推理：概率性、泛化强、语义理解
- 融合推理：符号引导 + LLM增强 + 反馈优化
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Engine, Parser, Memory, KnowledgeGraph, ToolRegistry, ActionType, Intent
from xiulian.api import Result


class FusionStrategy(Enum):
    SYMBOLIC_FIRST = "symbolic_first"
    LLM_FIRST = "llm_first"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class FusionResult:
    final_answer: Any
    symbolic_result: Optional[Result] = None
    llm_results: List[Dict] = field(default_factory=list)
    confidence: float = 1.0
    explanation: str = ""
    latency_ms: float = 0.0
    method_used: str = "fusion"
    metadata: Dict = field(default_factory=dict)


class OllamaClient:
    """ollama API客户端"""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 30.0):
        self.host = host
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.cache_size = 100
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Content-Type": "application/json"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def generate(self, model: str, prompt: str, system: str = None, 
                       temperature: float = 0.7, max_tokens: int = 500) -> Dict:
        cache_key = hashlib.md5(f"{model}:{prompt}:{temperature}".encode()).hexdigest()
        if cache_key in self.cache:
            content, _ = self.cache[cache_key]
            return {"response": content, "cached": True, "model": model}
        
        session = await self._get_session()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system
        
        try:
            async with session.post(f"{self.host}/api/generate", json=payload) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}", "model": model}
                data = await resp.json()
                
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[cache_key] = (data.get("response", ""), time.time())
                
                return {
                    "response": data.get("response", ""),
                    "model": model,
                    "total_duration": data.get("total_duration", 0),
                    "cached": False
                }
        except asyncio.TimeoutError:
            return {"error": "Timeout", "model": model}
        except Exception as e:
            return {"error": str(e), "model": model}
    
    async def chat(self, model: str, messages: List[Dict], temperature: float = 0.7) -> Dict:
        session = await self._get_session()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        try:
            async with session.post(f"{self.host}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}", "model": model}
                data = await resp.json()
                return {
                    "response": data.get("message", {}).get("content", ""),
                    "model": model,
                    "total_duration": data.get("total_duration", 0)
                }
        except Exception as e:
            return {"error": str(e), "model": model}
    
    async def list_models(self) -> List[Dict]:
        session = await self._get_session()
        try:
            async with session.get(f"{self.host}/api/tags") as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("models", [])
        except Exception as e:
            return []


class TraditionalMethodsWrapper:
    """传统AI方法包装器"""
    
    def __init__(self):
        self.methods = {}
        self._load_methods()
    
    def _load_methods(self):
        try:
            from non_dl_ai_benchmark.METHODS.01_symbolic.expert_system import ExpertSystem, CaseBasedReasoning
            self.methods["expert_system"] = ExpertSystem()
            self.methods["cbr"] = CaseBasedReasoning()
        except ImportError:
            pass
        
        try:
            from non_dl_ai_benchmark.METHODS.03_evolutionary.genetic_algorithm import GeneticAlgorithm
            self.methods["ga"] = None
        except ImportError:
            pass
        
        try:
            from non_dl_ai_benchmark.METHODS.05_bayesian.bayesian_networks import NaiveBayesClassifier
            self.methods["bayes"] = NaiveBayesClassifier()
        except ImportError:
            pass
        
        try:
            from non_dl_ai_benchmark.METHODS.06_decision_tree.decision_tree import ID3DecisionTree
            self.methods["decision_tree"] = ID3DecisionTree()
        except ImportError:
            pass
    
    def classify_with_rules(self, features: Dict) -> Tuple[str, float]:
        if "bayes" in self.methods and self.methods["bayes"]:
            try:
                pred = self.methods["bayes"].predict(features)
                return pred, 0.85
            except Exception:
                pass
        return "unknown", 0.0
    
    def optimize_with_ga(self, fitness_func, gene_length: int = 10, 
                         generations: int = 20) -> Tuple[List, float]:
        try:
            from non_dl_ai_benchmark.METHODS.03_evolutionary.genetic_algorithm import GeneticAlgorithm
            ga = GeneticAlgorithm(fitness_func=fitness_func, gene_length=gene_length, 
                                  population_size=30)
            best, fitness = ga.evolve(max_generations=generations)
            return best, fitness
        except Exception:
            return [], 0.0
    
    def retrieve_similar_case(self, problem: Dict) -> Tuple[Optional[Dict], float]:
        if "cbr" in self.methods and self.methods["cbr"]:
            try:
                similar = self.methods["cbr"].retrieve(problem, top_k=1)
                if similar:
                    return similar[0][0], similar[0][1]
            except Exception:
                pass
        return None, 0.0


class FusionEngine:
    """融合范式引擎核心
    
    整合三种范式：
    1. 符号推理 (XiuLian) - 快速、确定性
    2. 传统AI方法 - 专家系统、遗传算法、贝叶斯等
    3. LLM推理 (ollama) - 深度语义理解
    """
    
    def __init__(self, 
                 llm_models: List[str] = None,
                 strategy: FusionStrategy = FusionStrategy.HYBRID,
                 confidence_threshold: float = 0.8):
        self.xiulian_engine = Engine()
        self.ollama_client = OllamaClient()
        self.traditional_methods = TraditionalMethodsWrapper()
        
        self.llm_models = llm_models or ["qwen3:14b", "gemma3:4b", "granite4:350m"]
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        
        self.process_history: List[Dict] = []
        self.rule_updates: List[Dict] = []
        self.prompt_templates: Dict[str, str] = {
            "tool_call": "分析以下指令，提取意图和参数：\n{input}\n输出格式：intent=X, params={Y}",
            "question": "请回答问题：{input}\n要求简洁准确。",
            "classification": "根据特征{features}进行分类，类别包括：{categories}",
            "fallback": "请处理以下输入：{input}"
        }
        
        self._register_fusion_tools()
    
    def _register_fusion_tools(self):
        def llm_query(**kwargs):
            model = kwargs.get("model", self.llm_models[0])
            prompt = kwargs.get("prompt", kwargs.get("input", ""))
            return {"model": model, "prompt": prompt, "status": "queued"}
        
        def classify(**kwargs):
            features = kwargs.get("features", {})
            result, confidence = self.traditional_methods.classify_with_rules(features)
            return {"classification": result, "confidence": confidence}
        
        def optimize(**kwargs):
            target = kwargs.get("target", "max")
            iterations = kwargs.get("iterations", 20)
            return {"status": "optimization_queued", "target": target, "iterations": iterations}
        
        self.xiulian_engine.tools.register(ToolDef(
            name="llm_query",
            func=llm_query,
            description="查询LLM模型",
            params_schema={"model": "str", "prompt": "str"},
            category="llm"
        ))
        
        self.xiulian_engine.tools.register(ToolDef(
            name="classify",
            func=classify,
            description="传统方法分类",
            params_schema={"features": "dict"},
            category="traditional"
        ))
    
    def _symbolic_process(self, text: str) -> Result:
        return self.xiulian_engine.process(text)
    
    async def _llm_process(self, text: str, models: List[str] = None) -> List[Dict]:
        models = models or self.llm_models
        results = []
        
        prompt = self._build_llm_prompt(text)
        
        for model in models[:3]:
            result = await self.ollama_client.generate(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=300
            )
            results.append(result)
        
        return results
    
    def _build_llm_prompt(self, text: str, context: Dict = None) -> str:
        intent = self.xiulian_engine.parser.parse(text)
        
        if intent.action == ActionType.TOOL:
            template = self.prompt_templates["tool_call"]
            return template.format(input=text)
        elif intent.action == ActionType.QUESTION:
            template = self.prompt_templates["question"]
            return template.format(input=text)
        else:
            template = self.prompt_templates["fallback"]
            return template.format(input=text)
    
    def _fuse_results(self, symbolic_result: Result, llm_results: List[Dict], 
                      text: str) -> FusionResult:
        if symbolic_result.success and symbolic_result.data:
            symbolic_confidence = 1.0
            if isinstance(symbolic_result.data, dict):
                symbolic_confidence = symbolic_result.data.get("confidence", 1.0)
            
            if symbolic_confidence >= self.confidence_threshold:
                return FusionResult(
                    final_answer=symbolic_result.data,
                    symbolic_result=symbolic_result,
                    confidence=symbolic_confidence,
                    explanation="符号推理成功，置信度高",
                    method_used="symbolic",
                    latency_ms=symbolic_result.latency_ms
                )
        
        valid_llm_results = [r for r in llm_results if "error" not in r and r.get("response")]
        
        if valid_llm_results:
            responses = [r["response"] for r in valid_llm_results]
            
            fused_response = self._vote_responses(responses)
            
            return FusionResult(
                final_answer={"response": fused_response, "source": "llm_fusion"},
                symbolic_result=symbolic_result,
                llm_results=valid_llm_results,
                confidence=0.75,
                explanation=f"符号推理置信度不足，使用{len(valid_llm_results)}个LLM模型投票",
                method_used="llm_fusion",
                latency_ms=sum(r.get("total_duration", 0) for r in valid_llm_results) / 1e6
            )
        
        return FusionResult(
            final_answer={"error": "所有方法均失败"},
            confidence=0.0,
            explanation="符号推理和LLM推理均失败",
            method_used="failed",
            latency_ms=0
        )
    
    def _vote_responses(self, responses: List[str]) -> str:
        if not responses:
            return ""
        
        if len(responses) == 1:
            return responses[0]
        
        lengths = [len(r) for r in responses]
        best_idx = lengths.index(max(lengths))
        
        common_words = set()
        for resp in responses:
            words = set(re.findall(r'\w+', resp.lower()))
            common_words.update(words)
        
        if responses[0] in responses[1] or responses[1] in responses[0]:
            return max(responses, key=len)
        
        return responses[best_idx]
    
    async def process(self, text: str) -> FusionResult:
        start_time = time.time()
        
        symbolic_result = self._symbolic_process(text)
        
        llm_results = []
        if self.strategy == FusionStrategy.PARALLEL:
            llm_results = await self._llm_process(text)
        elif self.strategy == FusionStrategy.HYBRID:
            if symbolic_result.success:
                confidence = 1.0
                if isinstance(symbolic_result.data, dict):
                    confidence = symbolic_result.data.get("confidence", 1.0)
                if confidence < self.confidence_threshold:
                    llm_results = await self._llm_process(text)
        
        fused_result = self._fuse_results(symbolic_result, llm_results, text)
        fused_result.latency_ms = (time.time() - start_time) * 1000
        
        self.process_history.append({
            "input": text,
            "symbolic_success": symbolic_result.success,
            "llm_count": len(llm_results),
            "final_method": fused_result.method_used,
            "confidence": fused_result.confidence,
            "latency_ms": fused_result.latency_ms
        })
        
        return fused_result
    
    async def batch_process(self, texts: List[str]) -> List[FusionResult]:
        results = []
        for text in texts:
            result = await self.process(text)
            results.append(result)
        return results
    
    def update_rule(self, condition: Dict, action: Dict, confidence: float = 1.0):
        self.rule_updates.append({
            "condition": condition,
            "action": action,
            "confidence": confidence,
            "timestamp": time.time()
        })
        
        if hasattr(self.traditional_methods.methods.get("expert_system"), "add_rule"):
            es = self.traditional_methods.methods["expert_system"]
            es.add_rule(condition, action, priority=1, confidence=confidence)
    
    def optimize_prompt(self, task_type: str, new_template: str):
        self.prompt_templates[task_type] = new_template
    
    def get_stats(self) -> Dict:
        if not self.process_history:
            return {"total": 0}
        
        symbolic_success_rate = sum(1 for h in self.process_history if h["symbolic_success"]) / len(self.process_history)
        llm_usage_rate = sum(1 for h in self.process_history if h["llm_count"] > 0) / len(self.process_history)
        avg_latency = sum(h["latency_ms"] for h in self.process_history) / len(self.process_history)
        avg_confidence = sum(h["confidence"] for h in self.process_history) / len(self.process_history)
        
        return {
            "total_processed": len(self.process_history),
            "symbolic_success_rate": symbolic_success_rate,
            "llm_usage_rate": llm_usage_rate,
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "rule_updates": len(self.rule_updates)
        }
    
    def export_history(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "process_history": self.process_history,
                "rule_updates": self.rule_updates,
                "stats": self.get_stats()
            }, f, ensure_ascii=False, indent=2)


class MultiModelOrchestrator:
    """多模型编排器
    
    实现多LLM协同推理：
    - 并行查询多个模型
    - 结果投票与融合
    - 负载均衡
    """
    
    def __init__(self, models: List[str] = None, weights: Dict[str, float] = None):
        self.ollama = OllamaClient()
        self.models = models or ["qwen3:14b", "gemma3:4b", "granite4:3b"]
        self.weights = weights or {m: 1.0/len(self.models) for m in self.models}
        
        self.model_stats: Dict[str, Dict] = {m: {"calls": 0, "successes": 0, "avg_latency": 0} for m in self.models}
    
    async def query_all(self, prompt: str, system: str = None, 
                        temperature: float = 0.7) -> Dict:
        results = {}
        
        tasks = [
            self.ollama.generate(model, prompt, system, temperature)
            for model in self.models
        ]
        
        responses = await asyncio.gather(*tasks)
        
        for model, resp in zip(self.models, responses):
            results[model] = resp
            if "error" not in resp:
                self.model_stats[model]["successes"] += 1
            self.model_stats[model]["calls"] += 1
        
        return results
    
    def weighted_vote(self, results: Dict[str, Dict]) -> Tuple[str, float]:
        valid_results = {
            m: r["response"] for m, r in results.items() 
            if "error" not in r and r.get("response")
        }
        
        if not valid_results:
            return "", 0.0
        
        if len(valid_results) == 1:
            model, response = list(valid_results.items())[0]
            return response, self.weights.get(model, 0.5)
        
        scores = {}
        for model, response in valid_results.items():
            weight = self.weights.get(model, 0.5)
            quality = len(response) / 500
            scores[model] = weight * quality
        
        best_model = max(scores.keys(), key=lambda m: scores[m])
        
        return valid_results[best_model], scores[best_model]
    
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        rankings = []
        for model, stats in self.model_stats.items():
            if stats["calls"] > 0:
                success_rate = stats["successes"] / stats["calls"]
                rankings.append((model, success_rate))
        
        rankings.sort(key=lambda x: -x[1])
        return rankings


async def test_fusion_engine():
    engine = FusionEngine(
        llm_models=["granite4:350m", "gemma3:1b"],
        strategy=FusionStrategy.HYBRID
    )
    
    test_cases = [
        "调用echo msg=hello",
        "搜索人工智能",
        "什么是机器学习?",
        "计算 100*5",
    ]
    
    print("="*60)
    print("融合范式引擎测试")
    print("="*60)
    
    for case in test_cases:
        print(f"\n输入: {case}")
        result = await engine.process(case)
        print(f"方法: {result.method_used}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"延迟: {result.latency_ms:.2f}ms")
        print(f"结果: {result.final_answer}")
    
    stats = engine.get_stats()
    print(f"\n统计:")
    print(f"  总处理数: {stats['total_processed']}")
    print(f"  符号成功率: {stats['symbolic_success_rate']:.2%}")
    print(f"  平均置信度: {stats['avg_confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(test_fusion_engine())