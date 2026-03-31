# coding: utf-8
"""
全面测试与持续训练
包含更多测试用例，持续迭代提升
"""

import requests
import json
import time
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Parser, ActionType, KnowledgeGraph


class FusionStudentModel:
    """融合学生模型 - 整合规则、模板、知识和学习"""
    
    def __init__(self):
        self.parser = Parser()
        self.kg = KnowledgeGraph()
        self.rules = []
        self.templates = {}
        self.learned = {}
        self.categories = {}
        self.stats = {"rule": 0, "template": 0, "learned": 0, "symbolic": 0, "unknown": 0}
        self._initialize()
    
    def _initialize(self):
        self._setup_rules()
        self._setup_templates()
        self._setup_knowledge()
    
    def _setup_rules(self):
        rules_config = [
            (r'调用\s*echo\s+msg\s*=\s*(\w+)', lambda m: {"message": m.group(1)}),
            (r'调用\s*echo\s+message\s*=\s*(\w+)', lambda m: {"message": m.group(1)}),
            (r'调用\s*time', lambda m: {"time": time.strftime("%Y-%m-%d %H:%M:%S")}),
            (r'调用\s*random\s*min\s*=\s*(\d+)\s*max\s*=\s*(\d+)', 
             lambda m: {"number": (int(m.group(1)) + int(m.group(2))) // 2}),
            (r'计算\s*(\d+)\s*\+\s*(\d+)', lambda m: {"result": int(m.group(1)) + int(m.group(2)), "op": "+"}),
            (r'计算\s*(\d+)\s*\-\s*(\d+)', lambda m: {"result": int(m.group(1)) - int(m.group(2)), "op": "-"}),
            (r'计算\s*(\d+)\s*\*\s*(\d+)', lambda m: {"result": int(m.group(1)) * int(m.group(2)), "op": "*"}),
            (r'计算\s*(\d+)\s*/\s*(\d+)', lambda m: {"result": int(m.group(1)) // int(m.group(2)), "op": "/"}),
            (r'搜索\s*(.+)', lambda m: {"query": m.group(1).strip(), "action": "search"}),
            (r'查找\s*(.+)', lambda m: {"query": m.group(1).strip(), "action": "search"}),
            (r'打开\s*(https?://\S+)', lambda m: {"url": m.group(1), "action": "web"}),
        ]
        
        for pattern, handler in rules_config:
            self.rules.append((re.compile(pattern, re.I), handler, 1.0))
    
    def _setup_templates(self):
        self.templates = {
            "ai": "人工智能(AI)是模拟人类智能的技术领域，包括机器学习、深度学习、自然语言处理等分支。AI旨在让计算机执行需要人类智能的任务。",
            "ml": "机器学习是AI的核心分支，通过算法让计算机从数据中学习模式并做出预测。主要类型包括监督学习、无监督学习和强化学习。",
            "dl": "深度学习使用多层神经网络进行学习和特征提取，是机器学习的高级形式。主要应用于图像识别、语音识别和自然语言处理。",
            "nlp": "自然语言处理(NLP)让计算机理解、解释和生成人类语言。应用包括机器翻译、情感分析、问答系统等。",
            "python": "Python是一种高级编程语言，因其简洁的语法和丰富的AI/ML库而广泛应用于人工智能开发。",
            "neural": "神经网络是模拟人脑神经元连接的计算模型，由输入层、隐藏层和输出层组成，是深度学习的基础。",
            "transformer": "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于NLP任务，如GPT、BERT等模型。",
            "llm": "大语言模型(LLM)是基于Transformer的超大规模语言模型，如GPT-4、Claude等，具有强大的语言理解和生成能力。",
        }
        
        self.categories = {
            "ai": ["人工智能", "ai", "AI"],
            "ml": ["机器学习", "machine learning", "ML"],
            "dl": ["深度学习", "deep learning", "DL", "神经网络"],
            "nlp": ["nlp", "NLP", "自然语言处理", "自然语言"],
            "python": ["python", "Python", "编程"],
            "neural": ["神经网络", "neural network"],
            "transformer": ["transformer", "Transformer", "注意力", "transform"],
            "llm": ["llm", "LLM", "大语言模型", "大模型"],
        }
    
    def _setup_knowledge(self):
        entities = [
            ("AI", "concept"), ("人工智能", "concept"),
            ("机器学习", "concept"), ("ML", "concept"),
            ("深度学习", "concept"), ("DL", "concept"),
            ("NLP", "concept"), ("自然语言处理", "concept"),
            ("Python", "language"),
            ("神经网络", "technology"),
            ("Transformer", "architecture"),
            ("LLM", "model_type"), ("大语言模型", "model_type"),
            ("GPT", "model"), ("BERT", "model"), ("Claude", "model"),
        ]
        for name, etype in entities:
            self.kg.add_entity(name, etype)
        
        relations = [
            ("深度学习", "机器学习", "is_subtype_of"),
            ("机器学习", "人工智能", "is_subtype_of"),
            ("NLP", "人工智能", "is_application_of"),
            ("Transformer", "神经网络", "uses"),
            ("GPT", "Transformer", "based_on"),
            ("BERT", "Transformer", "based_on"),
            ("LLM", "Transformer", "based_on"),
        ]
        for src, tgt, rel in relations:
            self.kg.add_relation(src, tgt, rel)
    
    def process(self, text: str) -> Tuple[Dict, str, float]:
        for pattern, handler, conf in self.rules:
            match = pattern.search(text)
            if match:
                self.stats["rule"] += 1
                return handler(match), "rule", conf
        
        text_lower = text.lower()
        has_question = any(w in text_lower for w in ["什么", "?", "？", "如何", "怎么", "为什么", "what", "how", "why"])
        
        for cat, keywords in self.categories.items():
            if any(kw in text_lower for kw in keywords) and has_question:
                if cat in self.templates:
                    self.stats["template"] += 1
                    return {"response": self.templates[cat]}, "template", 0.9
        
        if text in self.learned:
            self.stats["learned"] += 1
            return {"response": self.learned[text]}, "learned", 0.8
        
        intent = self.parser.parse(text)
        if intent.action != ActionType.UNKNOWN:
            self.stats["symbolic"] += 1
            return {"intent": intent.action.value, "entities": intent.entities}, "symbolic", 0.7
        
        for q, a in self.learned.items():
            if self._similar(text, q) > 0.6:
                self.stats["learned"] += 1
                return {"response": a, "similar_to": q}, "learned", 0.6
        
        self.stats["unknown"] += 1
        return {"response": "我需要学习这个问题的答案", "need_learning": True}, "unknown", 0.0
    
    def _similar(self, t1: str, t2: str) -> float:
        w1, w2 = set(t1.lower().split()), set(t2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)
    
    def learn(self, question: str, answer: str, confidence: float = 0.8):
        self.learned[question] = answer
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "learned": self.learned,
                "rules_count": len(self.rules),
                "templates_count": len(self.templates),
                "entities_count": self.kg.graph.number_of_nodes(),
                "stats": self.stats,
                "timestamp": time.time()
            }, f, ensure_ascii=False, indent=2)


class ComprehensiveBenchmark:
    """全面测试基准"""
    
    def __init__(self):
        self.tests = self._create_tests()
        self.results = []
    
    def _create_tests(self) -> List[Dict]:
        return [
            {"id": "tool_echo", "q": "调用echo msg=hello", "cat": "tool",
             "check": lambda r: r.get("message") == "hello"},
            {"id": "tool_time", "q": "调用time", "cat": "tool",
             "check": lambda r: "time" in r},
            {"id": "math_add", "q": "计算2+2", "cat": "math",
             "check": lambda r: r.get("result") == 4},
            {"id": "math_mul", "q": "计算10*5", "cat": "math",
             "check": lambda r: r.get("result") == 50},
            {"id": "math_sub", "q": "计算100-30", "cat": "math",
             "check": lambda r: r.get("result") == 70},
            {"id": "search_1", "q": "搜索人工智能", "cat": "search",
             "check": lambda r: r.get("action") == "search"},
            {"id": "web_1", "q": "打开 https://example.com", "cat": "web",
             "check": lambda r: r.get("action") == "web"},
            {"id": "qa_ai", "q": "什么是人工智能?", "cat": "qa",
             "check": lambda r: "人工智能" in r.get("response", "") or "AI" in r.get("response", "")},
            {"id": "qa_ml", "q": "什么是机器学习?", "cat": "qa",
             "check": lambda r: "机器学习" in r.get("response", "")},
            {"id": "qa_dl", "q": "什么是深度学习?", "cat": "qa",
             "check": lambda r: "深度学习" in r.get("response", "")},
            {"id": "qa_nlp", "q": "NLP是什么?", "cat": "qa",
             "check": lambda r: "NLP" in r.get("response", "") or "自然语言" in r.get("response", "")},
            {"id": "qa_python", "q": "Python是什么?", "cat": "qa",
             "check": lambda r: "Python" in r.get("response", "") or "编程" in r.get("response", "")},
            {"id": "qa_nn", "q": "什么是神经网络?", "cat": "qa",
             "check": lambda r: "神经网络" in r.get("response", "") or "神经" in r.get("response", "")},
            {"id": "qa_transformer", "q": "Transformer是什么?", "cat": "qa",
             "check": lambda r: "Transformer" in r.get("response", "") or "注意力" in r.get("response", "")},
            {"id": "qa_llm", "q": "什么是大语言模型?", "cat": "qa",
             "check": lambda r: "大语言" in r.get("response", "") or "LLM" in r.get("response", "")},
        ]
    
    def evaluate(self, student: FusionStudentModel) -> Tuple[float, Dict]:
        results = []
        passed = 0
        by_category = {}
        
        for test in self.tests:
            result, method, conf = student.process(test["q"])
            
            try:
                success = test["check"](result) and conf >= 0.7
            except:
                success = False
            
            results.append({
                "id": test["id"],
                "category": test["cat"],
                "success": success,
                "method": method,
                "confidence": conf
            })
            
            if success:
                passed += 1
            
            cat = test["cat"]
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "total": 0}
            by_category[cat]["total"] += 1
            if success:
                by_category[cat]["passed"] += 1
        
        self.results = results
        accuracy = passed / len(self.tests)
        
        return accuracy, {
            "passed": passed,
            "total": len(self.tests),
            "by_category": by_category,
            "details": results
        }


class TeacherPool:
    def __init__(self):
        self.teachers = [
            {"name": "granite4:350m", "weight": 0.4, "calls": 0},
            {"name": "granite4:1b", "weight": 0.3, "calls": 0},
            {"name": "gemma3:1b", "weight": 0.2, "calls": 0},
            {"name": "gemma3:4b", "weight": 0.1, "calls": 0},
        ]
        self.idx = 0
    
    def get_next(self) -> str:
        t = self.teachers[self.idx]["name"]
        self.idx = (self.idx + 1) % len(self.teachers)
        return t
    
    def query(self, model: str, prompt: str, timeout: int = 30) -> Optional[str]:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": 150, "temperature": 0.7}},
                timeout=timeout
            )
            if resp.status_code == 200:
                for t in self.teachers:
                    if t["name"] == model:
                        t["calls"] += 1
                return resp.json().get("response", "")
        except:
            pass
        return None
    
    def get_stats(self) -> Dict:
        return {t["name"]: t["calls"] for t in self.teachers}


def main():
    print("="*70)
    print("Fusion Paradigm Model - Comprehensive Training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    output_dir = Path("fusion_paradigm/trained_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student = FusionStudentModel()
    benchmark = ComprehensiveBenchmark()
    teachers = TeacherPool()
    
    print(f"\nModel initialized:")
    print(f"  Rules: {len(student.rules)}")
    print(f"  Templates: {len(student.templates)}")
    print(f"  Knowledge entities: {student.kg.graph.number_of_nodes()}")
    
    print("\n" + "-"*70)
    print("Baseline Evaluation")
    print("-"*70)
    
    acc, details = benchmark.evaluate(student)
    print(f"\nOverall Accuracy: {acc:.1%} ({details['passed']}/{details['total']})")
    
    print("\nBy Category:")
    for cat, stats in details["by_category"].items():
        cat_acc = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {cat_acc:.1%} ({stats['passed']}/{stats['total']})")
    
    if acc >= 0.90:
        print(f"\n[TARGET] Already at {acc:.1%} - Excellent!")
    
    print("\n" + "-"*70)
    print("Continuous Training (Ctrl+C to stop)")
    print("-"*70)
    
    questions_to_learn = [
        "GPT和BERT的区别是什么?",
        "如何评估机器学习模型的性能?",
        "什么是过拟合和欠拟合?",
        "卷积神经网络和循环神经网络的区别?",
        "什么是强化学习?",
        "NLP中的注意力机制是什么?",
    ]
    
    max_iterations = 30
    target_accuracy = 0.90
    
    try:
        for i in range(1, max_iterations + 1):
            if acc >= target_accuracy:
                print(f"\n[TARGET] Accuracy {acc:.1%} reached target {target_accuracy:.0%}!")
                break
            
            print(f"\n[Iteration {i}/{max_iterations}] Current: {acc:.1%}")
            
            for q in questions_to_learn:
                result, method, conf = student.process(q)
                
                if conf < 0.8:
                    teacher = teachers.get_next()
                    prompt = f"请用中文简洁回答：{q}"
                    answer = teachers.query(teacher, prompt)
                    
                    if answer:
                        student.learn(q, answer)
                        print(f"  Learned from {teacher}: {q[:30]}...")
            
            acc, details = benchmark.evaluate(student)
            
            if i % 5 == 0:
                student.save(str(output_dir / f"model_iter_{i}.json"))
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    print("\n" + "-"*70)
    print("Final Results")
    print("-"*70)
    
    acc, details = benchmark.evaluate(student)
    print(f"\nFinal Accuracy: {acc:.1%}")
    
    print("\nDetailed Results:")
    for r in details["details"]:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['method']} ({r['confidence']:.0%})")
    
    student.save(str(output_dir / "model_final.json"))
    
    training_log = {
        "final_accuracy": acc,
        "iterations": i if 'i' in dir() else 0,
        "teacher_calls": teachers.get_stats(),
        "learned_count": len(student.learned),
        "stats": student.stats,
        "timestamp": datetime.now().isoformat(),
        "category_results": details["by_category"]
    }
    
    with open(output_dir / "training_log.json", 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print(f"Training Complete - Final Accuracy: {acc:.1%}")
    print(f"Learned: {len(student.learned)} new answers")
    print(f"Teacher calls: {sum(teachers.get_stats().values())}")
    print("="*70)
    
    return acc


if __name__ == "__main__":
    main()