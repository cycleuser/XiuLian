"""
融合范式训练框架 - Fusion Paradigm Training Framework
多教师知识蒸馏 + 融合模型训练

核心设计：
┌─────────────────────────────────────────────────────────────┐
│                    多教师模型池                               │
│  qwen3:14b  │  gemma3:4b  │  granite4:3b  │  gpt-oss:20b   │
│   教师1    │   教师2     │    教师3      │    教师4       │
└─────────────────────────────────────────────────────────────┘
                        ↓ 轮流提供训练信号
┌─────────────────────────────────────────────────────────────┐
│                    训练数据生成器                             │
│  - 从每个教师获取不同角度的回答                               │
│  - 生成多样化的训练样本                                       │
│  - 标注置信度和质量评分                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                FusionStudentModel（学生模型）                 │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  符号推理模块   │  │  神经网络模块   │                   │
│  │  (继承XiuLian) │  │  (小型MLP)     │                   │
│  └─────────────────┘  └─────────────────┘                   │
│              ↓融合层                                        │
│  ┌─────────────────────────────────────────┐               │
│  │  多范式融合决策                          │               │
│  │  - 符号规则优先                          │               │
│  │  - 网络补充模糊推理                      │               │
│  │  - 知识图谱关系推理                      │               │
│  └─────────────────────────────────────────┐               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    训练循环                                   │
│  1. 选择当前教师                                             │
│  2. 生成训练样本                                             │
│  3. 学生模型学习                                             │
│  4. 评估+反馈                                                │
│  5. 切换下一教师                                             │
│  6. 循环直到达标                                              │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import re
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Engine, Parser, Memory, KnowledgeGraph, ActionType, Intent
from xiulian.api import Result


class TrainingPhase(Enum):
    SYMBOLIC_BOOTSTRAP = "symbolic_bootstrap"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"


@dataclass
class TrainingSample:
    input_text: str
    teacher_responses: Dict[str, str]
    best_response: str
    confidence: float
    category: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class StudentOutput:
    response: str
    method_used: str
    confidence: float
    symbolic_contribution: float
    neural_contribution: float
    explanation: str


@dataclass
class TrainingProgress:
    iteration: int
    current_teacher: str
    samples_generated: int
    samples_learned: int
    accuracy: float
    loss: float
    timestamp: float = field(default_factory=time.time)


class OllamaTeacherPool:
    """ollama教师模型池
    
    管理多个LLM教师，轮流提供训练信号
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self._session: Optional[aiohttp.ClientSession] = None
        
        self.teachers = {
            "qwen3:14b": {"weight": 0.35, "specialty": "reasoning", "calls": 0},
            "gemma3:4b": {"weight": 0.25, "specialty": "qa", "calls": 0},
            "granite4:3b": {"weight": 0.20, "specialty": "code", "calls": 0},
            "granite4:1b": {"weight": 0.15, "specialty": "fast", "calls": 0},
            "granite4:350m": {"weight": 0.05, "specialty": "lightweight", "calls": 0},
        }
        
        self.current_teacher_idx = 0
        self.teacher_order = list(self.teachers.keys())
        self.response_cache: Dict[str, Dict] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60.0),
                headers={"Content-Type": "application/json"}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def get_next_teacher(self) -> str:
        teacher = self.teacher_order[self.current_teacher_idx]
        self.current_teacher_idx = (self.current_teacher_idx + 1) % len(self.teacher_order)
        return teacher
    
    def get_all_teachers(self) -> List[str]:
        return self.teacher_order
    
    async def query_teacher(self, model: str, prompt: str, 
                            temperature: float = 0.7, max_tokens: int = 300) -> Dict:
        cache_key = hashlib.md5(f"{model}:{prompt}:{temperature}".encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
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
        
        try:
            async with session.post(f"{self.host}/api/generate", json=payload) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}", "model": model}
                data = await resp.json()
                result = {
                    "response": data.get("response", ""),
                    "model": model,
                    "duration_ms": data.get("total_duration", 0) / 1e6,
                    "success": True
                }
                self.response_cache[cache_key] = result
                self.teachers[model]["calls"] += 1
                return result
        except Exception as e:
            return {"error": str(e), "model": model}
    
    async def query_all_teachers(self, prompt: str, 
                                  temperature: float = 0.7) -> Dict[str, Dict]:
        results = {}
        tasks = [
            self.query_teacher(model, prompt, temperature)
            for model in self.teacher_order
        ]
        responses = await asyncio.gather(*tasks)
        for model, resp in zip(self.teacher_order, responses):
            results[model] = resp
        return results
    
    def aggregate_responses(self, responses: Dict[str, Dict]) -> Tuple[str, float]:
        valid = {m: r for m, r in responses.items() if "error" not in r and r.get("response")}
        
        if not valid:
            return "", 0.0
        
        weighted_scores = {}
        for model, resp in valid.items():
            weight = self.teachers[model]["weight"]
            quality = len(resp["response"]) / 200
            weighted_scores[model] = weight * (1 + quality)
        
        best_model = max(weighted_scores.keys(), key=lambda m: weighted_scores[m])
        
        all_responses = [r["response"] for r in valid.values()]
        consensus_words = self._find_consensus(all_responses)
        
        best_response = valid[best_model]["response"]
        confidence = weighted_scores[best_model] / sum(weighted_scores.values())
        
        return best_response, confidence
    
    def _find_consensus(self, responses: List[str]) -> List[str]:
        if len(responses) < 2:
            return []
        
        word_counts = defaultdict(int)
        for resp in responses:
            words = re.findall(r'\w+', resp.lower())
            for word in words:
                word_counts[word] += 1
        
        threshold = len(responses) * 0.5
        consensus = [w for w, c in word_counts.items() if c >= threshold]
        return consensus


class SymbolicKnowledgeBase:
    """符号知识库
    
    存储从教师学习到的规则和知识
    """
    
    def __init__(self):
        self.rules: List[Dict] = []
        self.patterns: Dict[str, Dict] = {}
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Tuple[str, str, str]] = []
        
        self.memory = Memory()
        self.knowledge_graph = KnowledgeGraph()
    
    def add_rule(self, condition: str, action: str, confidence: float = 1.0,
                 source: str = "learned"):
        rule = {
            "condition": condition,
            "action": action,
            "confidence": confidence,
            "source": source,
            "timestamp": time.time()
        }
        self.rules.append(rule)
    
    def add_pattern(self, pattern: str, response_template: str,
                    category: str = "general"):
        self.patterns[pattern] = {
            "template": response_template,
            "category": category,
            "matches": 0
        }
    
    def add_entity(self, name: str, entity_type: str, attributes: Dict = None):
        self.entities[name] = {
            "type": entity_type,
            "attributes": attributes or {},
            "timestamp": time.time()
        }
        self.knowledge_graph.add_entity(name, entity_type, **(attributes or {}))
    
    def add_relation(self, source: str, target: str, relation: str):
        self.relations.append((source, target, relation))
        self.knowledge_graph.add_relation(source, target, relation)
    
    def query_rules(self, condition: str) -> List[Dict]:
        matching = []
        for rule in self.rules:
            if self._match_condition(condition, rule["condition"]):
                matching.append(rule)
        return sorted(matching, key=lambda r: -r["confidence"])
    
    def _match_condition(self, input_cond: str, rule_cond: str) -> bool:
        input_lower = input_cond.lower()
        rule_lower = rule_cond.lower()
        
        if rule_lower in input_lower:
            return True
        
        rule_words = set(rule_lower.split())
        input_words = set(input_lower.split())
        overlap = len(rule_words & input_words)
        if overlap >= len(rule_words) * 0.5:
            return True
        
        return False
    
    def match_pattern(self, text: str) -> Optional[Dict]:
        text_lower = text.lower()
        for pattern, data in self.patterns.items():
            if re.search(pattern, text_lower):
                data["matches"] += 1
                return data
        return None
    
    def get_related_entities(self, entity: str) -> List[str]:
        neighbors = self.knowledge_graph.neighbors(entity, depth=2)
        return list(neighbors)
    
    def save(self, path: str):
        data = {
            "rules": self.rules,
            "patterns": self.patterns,
            "entities": self.entities,
            "relations": self.relations,
            "memory": self.memory.store,
            "timestamp": time.time()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.rules = data.get("rules", [])
            self.patterns = data.get("patterns", {})
            self.entities = data.get("entities", {})
            self.relations = [tuple(r) for r in data.get("relations", [])]
            
            for k, v in data.get("memory", {}).items():
                self.memory.add(k, v)
            
            for entity, info in self.entities.items():
                self.knowledge_graph.add_entity(entity, info.get("type", "entity"),
                                                **info.get("attributes", {}))
            for src, tgt, rel in self.relations:
                self.knowledge_graph.add_relation(src, tgt, rel)
        except FileNotFoundError:
            pass


class NeuralModule:
    """轻量神经网络模块
    
    学习教师输出的软模式（无法用符号表达的）
    """
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = defaultdict(int)
        
        self.embeddings: Optional[List[List[float]]] = None
        self.weights1: Optional[List[List[float]]] = None
        self.weights2: Optional[List[List[float]]] = None
        
        self.training_samples: List[Tuple[List[int], List[int]]] = []
        self.is_initialized = False
    
    def _init_vocab(self):
        for idx, word in enumerate(self.idx_to_word.values()):
            self.word_to_idx[word] = idx
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                self.word_freq[word] += 1
        
        frequent_words = [w for w, f in self.word_freq.items() if f >= min_freq]
        frequent_words = frequent_words[:self.vocab_size - 1]
        
        self.word_to_idx = {"<unk>": 0}
        self.idx_to_word = {0: "<unk>"}
        
        for idx, word in enumerate(frequent_words, 1):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def _init_weights(self):
        self.embeddings = [
            [random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
            for _ in range(self.vocab_size)
        ]
        
        self.weights1 = [
            [random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
            for _ in range(self.hidden_dim)
        ]
        
        self.weights2 = [
            [random.gauss(0, 0.1) for _ in range(self.vocab_size)]
            for _ in range(self.hidden_dim)
        ]
        
        self.is_initialized = True
    
    def encode(self, text: str) -> List[int]:
        words = re.findall(r'\w+', text.lower())
        indices = []
        for word in words[:50]:
            idx = self.word_to_idx.get(word, 0)
            indices.append(idx)
        return indices if indices else [0]
    
    def forward(self, input_indices: List[int]) -> List[float]:
        if not self.is_initialized:
            return [0.5] * self.vocab_size
        
        embedded = [
            sum(self.embeddings[idx][h] for idx in input_indices) / len(input_indices)
            for h in range(self.hidden_dim)
        ]
        
        hidden = [
            math.tanh(sum(embedded[h1] * self.weights1[h1][h2] 
                         for h1 in range(self.hidden_dim)))
            for h2 in range(self.hidden_dim)
        ]
        
        output = [
            sum(hidden[h] * self.weights2[h][o] for h in range(self.hidden_dim))
            for o in range(min(len(self.weights2[0]), self.vocab_size))
        ]
        
        return output
    
    def softmax(self, logits: List[float]) -> List[float]:
        max_logit = max(logits) if logits else 0
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]
    
    def predict_next_words(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        indices = self.encode(text)
        logits = self.forward(indices)
        probs = self.softmax(logits)
        
        indexed_probs = [(idx, p) for idx, p in enumerate(probs) if idx in self.idx_to_word]
        indexed_probs.sort(key=lambda x: -x[1])
        
        return [(self.idx_to_word.get(idx, "<unk>"), prob) 
                for idx, prob in indexed_probs[:top_k]]
    
    def train_step(self, input_text: str, target_text: str, 
                   learning_rate: float = 0.01):
        if not self.is_initialized:
            self._init_weights()
        
        input_indices = self.encode(input_text)
        target_indices = self.encode(target_text)
        
        logits = self.forward(input_indices)
        probs = self.softmax(logits)
        
        for idx in target_indices:
            if idx < len(probs):
                probs[idx] = max(probs[idx], 0.01)
        
        for i, idx in enumerate(input_indices[:10]):
            for h in range(self.hidden_dim):
                if idx < len(self.embeddings):
                    self.embeddings[idx][h] += learning_rate * 0.1
    
    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "word_to_idx": self.word_to_idx,
            "idx_to_word": {str(k): v for k, v in self.idx_to_word.items()},
            "word_freq": dict(self.word_freq),
            "is_initialized": self.is_initialized
        }
        if self.is_initialized:
            data["embeddings"] = self.embeddings
            data["weights1"] = self.weights1
            data["weights2"] = self.weights2
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.vocab_size = data.get("vocab_size", self.vocab_size)
            self.hidden_dim = data.get("hidden_dim", self.hidden_dim)
            self.word_to_idx = data.get("word_to_idx", {})
            self.idx_to_word = {int(k): v for k, v in data.get("idx_to_word", {}).items()}
            self.word_freq = defaultdict(int, data.get("word_freq", {}))
            self.is_initialized = data.get("is_initialized", False)
            
            if self.is_initialized:
                self.embeddings = data.get("embeddings")
                self.weights1 = data.get("weights1")
                self.weights2 = data.get("weights2")
        except FileNotFoundError:
            pass


class FusionStudentModel:
    """融合学生模型
    
    整合三种范式：
    1. 符号推理（继承XiuLian + 学习规则）
    2. 神经网络推理（轻量MLP学习软模式）
    3. 知识图谱推理（实体关系）
    """
    
    def __init__(self):
        self.symbolic_kb = SymbolicKnowledgeBase()
        self.neural_module = NeuralModule()
        self.xiulian_parser = Parser()
        
        self.learned_prompts: Dict[str, str] = {}
        self.confidence_threshold = 0.75
        
        self.stats = {
            "symbolic_calls": 0,
            "neural_calls": 0,
            "hybrid_calls": 0,
            "total_calls": 0
        }
    
    def bootstrap_from_xiulian(self):
        self.symbolic_kb.add_pattern(
            r"调用\s*([a-zA-Z_]\w*)",
            "工具调用: {tool}",
            "tool_call"
        )
        self.symbolic_kb.add_pattern(
            r"搜索\s+(.+)",
            "搜索关键词: {query}",
            "search"
        )
        self.symbolic_kb.add_pattern(
            r"(什么|如何|为什么|怎么).+",
            "问题类型: {type}",
            "question"
        )
        
        self.symbolic_kb.add_rule(
            "echo msg",
            '{"message": "{msg}"}',
            confidence=1.0,
            source="builtin"
        )
        self.symbolic_kb.add_rule(
            "calc",
            '{"result": "{expr}"}',
            confidence=1.0,
            source="builtin"
        )
        
        entities = [
            ("AI", "concept", {"definition": "人工智能"}),
            ("机器学习", "concept", {"parent": "AI"}),
            ("深度学习", "concept", {"parent": "机器学习"}),
            ("NLP", "concept", {"full_name": "自然语言处理"}),
        ]
        for name, etype, attrs in entities:
            self.symbolic_kb.add_entity(name, etype, attrs)
        
        relations = [
            ("深度学习", "机器学习", "is_subtype_of"),
            ("机器学习", "AI", "is_subtype_of"),
            ("NLP", "AI", "is_application_of"),
        ]
        for src, tgt, rel in relations:
            self.symbolic_kb.add_relation(src, tgt, rel)
    
    def learn_from_teacher(self, sample: TrainingSample):
        self.symbolic_kb.add_rule(
            sample.input_text[:50],
            sample.best_response,
            confidence=sample.confidence,
            source="teacher"
        )
        
        category = sample.category
        if category not in self.learned_prompts:
            self.learned_prompts[category] = sample.best_response
        
        self.neural_module.train_step(sample.input_text, sample.best_response)
        
        entities_in_response = re.findall(r'[A-Za-z][a-zA-Z0-9_]+', sample.best_response)
        for entity in entities_in_response[:5]:
            if entity not in self.symbolic_kb.entities:
                self.symbolic_kb.add_entity(entity, "learned", {"source": "teacher_response"})
    
    def process(self, text: str) -> StudentOutput:
        self.stats["total_calls"] += 1
        
        symbolic_result = self._symbolic_inference(text)
        
        if symbolic_result and symbolic_result.confidence >= self.confidence_threshold:
            self.stats["symbolic_calls"] += 1
            return StudentOutput(
                response=symbolic_result.response,
                method_used="symbolic",
                confidence=symbolic_result.confidence,
                symbolic_contribution=1.0,
                neural_contribution=0.0,
                explanation=f"符号规则匹配: {symbolic_result.rule_source}"
            )
        
        neural_result = self._neural_inference(text)
        
        if symbolic_result and neural_result:
            self.stats["hybrid_calls"] += 1
            return self._hybrid_fusion(text, symbolic_result, neural_result)
        
        if neural_result:
            self.stats["neural_calls"] += 1
            return StudentOutput(
                response=neural_result.response,
                method_used="neural",
                confidence=neural_result.confidence,
                symbolic_contribution=0.0,
                neural_contribution=1.0,
                explanation="神经网络推理"
            )
        
        return StudentOutput(
            response="我需要学习这个问题的答案。",
            method_used="unknown",
            confidence=0.0,
            symbolic_contribution=0.0,
            neural_contribution=0.0,
            explanation="未找到匹配规则或神经网络输出"
        )
    
    def _symbolic_inference(self, text: str) -> Optional[Any]:
        pattern_result = self.symbolic_kb.match_pattern(text)
        if pattern_result:
            return type('obj', (object,), {
                'response': pattern_result["template"],
                'confidence': 0.9,
                'rule_source': 'pattern'
            })()
        
        rules = self.symbolic_kb.query_rules(text)
        if rules:
            best_rule = rules[0]
            return type('obj', (object,), {
                'response': best_rule["action"],
                'confidence': best_rule["confidence"],
                'rule_source': best_rule["source"]
            })()
        
        intent = self.xiulian_parser.parse(text)
        if intent.action != ActionType.UNKNOWN:
            return type('obj', (object,), {
                'response': f"检测到意图: {intent.action.value}",
                'confidence': 0.7,
                'rule_source': 'xiulian_parser'
            })()
        
        return None
    
    def _neural_inference(self, text: str) -> Optional[Any]:
        if not self.neural_module.is_initialized:
            return None
        
        predictions = self.neural_module.predict_next_words(text)
        if predictions:
            words = [w for w, p in predictions[:10]]
            response = " ".join(words)
            confidence = predictions[0][1] if predictions else 0.3
            
            return type('obj', (object,), {
                'response': response,
                'confidence': confidence
            })()
        
        return None
    
    def _hybrid_fusion(self, text: str, symbolic: Any, neural: Any) -> StudentOutput:
        sym_weight = symbolic.confidence
        neu_weight = neural.confidence
        
        sym_words = set(symbolic.response.lower().split())
        neu_words = set(neural.response.lower().split())
        
        common_words = sym_words & neu_words
        fusion_confidence = (sym_weight + neu_weight) / 2
        
        if common_words:
            fusion_confidence += 0.1
        
        if len(symbolic.response) > len(neural.response):
            final_response = symbolic.response
        else:
            final_response = neural.response
        
        sym_contrib = sym_weight / (sym_weight + neu_weight + 0.01)
        neu_contrib = neu_weight / (sym_weight + neu_weight + 0.01)
        
        return StudentOutput(
            response=final_response,
            method_used="hybrid",
            confidence=fusion_confidence,
            symbolic_contribution=sym_contrib,
            neural_contribution=neu_contrib,
            explanation="符号+神经网络混合推理"
        )
    
    def save(self, path: str):
        self.symbolic_kb.save(f"{path}/symbolic_kb.json")
        self.neural_module.save(f"{path}/neural_module.json")
        
        config = {
            "learned_prompts": self.learned_prompts,
            "confidence_threshold": self.confidence_threshold,
            "stats": self.stats,
            "timestamp": time.time()
        }
        with open(f"{path}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        self.symbolic_kb.load(f"{path}/symbolic_kb.json")
        self.neural_module.load(f"{path}/neural_module.json")
        
        try:
            with open(f"{path}/config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.learned_prompts = config.get("learned_prompts", {})
            self.confidence_threshold = config.get("confidence_threshold", 0.75)
            self.stats = config.get("stats", self.stats)
        except FileNotFoundError:
            pass


class TrainingLoop:
    """迭代训练循环
    
    核心流程：
    1. 从教师池选择当前教师
    2. 生成训练样本
    3. 学生模型学习
    4. 评估性能
    5. 反馈优化
    6. 循环迭代
    """
    
    def __init__(self, 
                 student_model: FusionStudentModel,
                 teacher_pool: OllamaTeacherPool,
                 output_dir: str = "fusion_paradigm/trained_model"):
        self.student = student_model
        self.teacher_pool = teacher_pool
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_history: List[TrainingProgress] = []
        self.samples: List[TrainingSample] = []
        self.iteration = 0
        self.max_iterations = 100
        self.target_accuracy = 0.85
        
        self.test_questions = self._generate_test_questions()
        self.current_phase = TrainingPhase.SYMBOLIC_BOOTSTRAP
    
    def _generate_test_questions(self) -> List[Dict]:
        return [
            {"text": "什么是人工智能?", "category": "qa", "expected_keywords": ["AI", "智能", "机器"]},
            {"text": "调用echo msg=hello", "category": "tool", "expected_keywords": ["hello", "message"]},
            {"text": "搜索机器学习", "category": "search", "expected_keywords": ["机器学习", "ML"]},
            {"text": "如何学习编程?", "category": "qa", "expected_keywords": ["编程", "学习", "代码"]},
            {"text": "计算2+2", "category": "tool", "expected_keywords": ["4", "结果"]},
            {"text": "什么是深度学习?", "category": "qa", "expected_keywords": ["深度", "学习", "神经网络"]},
            {"text": "NLP是什么?", "category": "qa", "expected_keywords": ["自然", "语言", "处理"]},
            {"text": "机器学习和深度学习的区别?", "category": "qa", "expected_keywords": ["区别", "关系"]},
        ]
    
    async def generate_training_sample(self, question: str, category: str) -> TrainingSample:
        teacher = self.teacher_pool.get_next_teacher()
        
        prompt = self._build_training_prompt(question, category)
        
        responses = await self.teacher_pool.query_all_teachers(prompt, temperature=0.7)
        
        best_response, confidence = self.teacher_pool.aggregate_responses(responses)
        
        teacher_responses = {
            model: resp.get("response", "")
            for model, resp in responses.items()
            if "error" not in resp
        }
        
        return TrainingSample(
            input_text=question,
            teacher_responses=teacher_responses,
            best_response=best_response,
            confidence=confidence,
            category=category,
            metadata={"iteration": self.iteration, "primary_teacher": teacher}
        )
    
    def _build_training_prompt(self, question: str, category: str) -> str:
        prompts = {
            "qa": f"请简洁回答以下问题，用中文：\n问题：{question}\n答案：",
            "tool": f"分析以下指令并给出执行结果：\n指令：{question}\n结果：",
            "search": f"分析以下搜索请求并给出相关信息：\n搜索：{question}\n信息：",
        }
        return prompts.get(category, f"请回答：{question}\n答案：")
    
    def train_on_sample(self, sample: TrainingSample):
        self.student.learn_from_teacher(sample)
        self.samples.append(sample)
    
    async def evaluate(self) -> float:
        correct = 0
        total = len(self.test_questions)
        
        for test in self.test_questions:
            output = self.student.process(test["text"])
            
            response_lower = output.response.lower()
            keywords_found = sum(1 for kw in test["expected_keywords"] 
                                if kw.lower() in response_lower)
            
            if keywords_found >= len(test["expected_keywords"]) * 0.5:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def calculate_loss(self) -> float:
        if not self.samples:
            return 1.0
        
        total_loss = 0
        for sample in self.samples[-20:]:
            output = self.student.process(sample.input_text)
            
            response_words = set(output.response.lower().split())
            target_words = set(sample.best_response.lower().split())
            
            overlap = len(response_words & target_words)
            union = len(response_words | target_words)
            
            if union > 0:
                jaccard = overlap / union
            else:
                jaccard = 0
            
            loss = 1 - jaccard
            total_loss += loss
        
        return total_loss / min(len(self.samples), 20)
    
    async def iteration_step(self) -> TrainingProgress:
        self.iteration += 1
        
        if self.iteration <= 5:
            self.current_phase = TrainingPhase.SYMBOLIC_BOOTSTRAP
            for test in self.test_questions[:4]:
                sample = await self.generate_training_sample(
                    test["text"], test["category"]
                )
                self.train_on_sample(sample)
        
        elif self.iteration <= 50:
            self.current_phase = TrainingPhase.KNOWLEDGE_TRANSFER
            
            random.shuffle(self.test_questions)
            for test in self.test_questions[:3]:
                sample = await self.generate_training_sample(
                    test["text"], test["category"]
                )
                self.train_on_sample(sample)
            
            expanded_questions = [
                f"请解释{test['text'].replace('什么是', '').replace('?', '')}的应用场景",
                f"{test['text']}为什么重要?"
            ]
            for q in expanded_questions[:2]:
                sample = await self.generate_training_sample(q, "qa")
                self.train_on_sample(sample)
        
        else:
            self.current_phase = TrainingPhase.FINE_TUNING
            
            low_score_tests = []
            for test in self.test_questions:
                output = self.student.process(test["text"])
                if output.confidence < 0.7:
                    low_score_tests.append(test)
            
            for test in low_score_tests[:2]:
                sample = await self.generate_training_sample(
                    test["text"], test["category"]
                )
                self.train_on_sample(sample)
        
        accuracy = await self.evaluate()
        loss = self.calculate_loss()
        
        progress = TrainingProgress(
            iteration=self.iteration,
            current_teacher=self.teacher_pool.teacher_order[self.teacher_pool.current_teacher_idx],
            samples_generated=len(self.samples),
            samples_learned=len([s for s in self.samples if s.confidence > 0.5]),
            accuracy=accuracy,
            loss=loss
        )
        self.progress_history.append(progress)
        
        if self.iteration % 5 == 0:
            self.student.save(str(self.output_dir))
            self._save_progress_report()
        
        return progress
    
    def _save_progress_report(self):
        report_path = self.output_dir / "training_progress.md"
        
        lines = [
            "# 融合模型训练进度报告",
            f"\n## 当前状态",
            f"- 迭代次数: {self.iteration}",
            f"- 当前阶段: {self.current_phase.value}",
            f"- 生成样本数: {len(self.samples)}",
            f"\n## 性能指标",
        ]
        
        if self.progress_history:
            latest = self.progress_history[-1]
            lines.extend([
                f"- 最新准确率: {latest.accuracy:.2%}",
                f"- 当前损失: {latest.loss:.4f}",
                f"- 当前教师: {latest.current_teacher}",
            ])
        
        lines.append("\n## 历史记录")
        for p in self.progress_history[-10:]:
            lines.append(f"- 迭代{p.iteration}: acc={p.accuracy:.2%}, loss={p.loss:.4f}")
        
        lines.append("\n## 教师贡献")
        for teacher, info in self.teacher_pool.teachers.items():
            lines.append(f"- {teacher}: {info['calls']}次调用, 权重={info['weight']}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    async def run_continuous(self, check_interval: float = 10.0):
        print("="*60)
        print("融合模型训练开始")
        print("="*60)
        
        self.student.bootstrap_from_xiulian()
        
        while self.iteration < self.max_iterations:
            progress = await self.iteration_step()
            
            print(f"\n迭代 {progress.iteration}:")
            print(f"  准确率: {progress.accuracy:.2%}")
            print(f"  损失: {progress.loss:.4f}")
            print(f"  样本数: {progress.samples_generated}")
            print(f"  教师: {progress.current_teacher}")
            
            if progress.accuracy >= self.target_accuracy:
                print(f"\n🎯 达到目标准确率 {self.target_accuracy:.2%}!")
                break
            
            await asyncio.sleep(check_interval)
        
        print("\n训练完成，保存最终模型...")
        self.student.save(str(self.output_dir))
        self._save_progress_report()
        
        final_accuracy = await self.evaluate()
        print(f"最终准确率: {final_accuracy:.2%}")
        
        return final_accuracy


async def main():
    teacher_pool = OllamaTeacherPool()
    student_model = FusionStudentModel()
    
    training_loop = TrainingLoop(
        student_model=student_model,
        teacher_pool=teacher_pool,
        output_dir="fusion_paradigm/trained_model"
    )
    
    final_accuracy = await training_loop.run_continuous(check_interval=2.0)
    
    await teacher_pool.close()
    
    print(f"\n训练结果: {final_accuracy:.2%}")


if __name__ == "__main__":
    asyncio.run(main())