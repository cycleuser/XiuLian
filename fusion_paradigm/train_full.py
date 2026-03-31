# coding: utf-8
"""
完整融合训练脚本
支持多教师轮流训练，持续迭代
"""

import requests
import json
import time
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Parser, Memory, KnowledgeGraph, ActionType


@dataclass
class TrainingSample:
    question: str
    answers: Dict[str, str]
    best_answer: str
    category: str
    confidence: float = 1.0


class FusionStudentModel:
    def __init__(self):
        self.parser = Parser()
        self.memory = Memory()
        self.kg = KnowledgeGraph()
        
        self.rules: List[Dict] = []
        self.patterns: Dict[str, callable] = {}
        self.learned_answers: Dict[str, str] = {}
        self.category_templates: Dict[str, str] = {}
        
        self.stats = {
            "rule_hits": 0,
            "pattern_hits": 0,
            "learned_hits": 0,
            "unknown_hits": 0,
            "total_queries": 0
        }
    
    def bootstrap(self):
        self._add_rule(r"调用\s*echo\s+msg\s*=\s*(\w+)", 
                       lambda m: {"message": m.group(1)}, "tool")
        self._add_rule(r"调用\s*echo\s+message\s*=\s*(\w+)", 
                       lambda m: {"message": m.group(1)}, "tool")
        self._add_rule(r"计算\s*(\d+)\s*\+\s*(\d+)", 
                       lambda m: {"result": int(m.group(1)) + int(m.group(2)), "expression": f"{m.group(1)}+{m.group(2)}"}, "math")
        self._add_rule(r"计算\s*(\d+)\s*\*\s*(\d+)", 
                       lambda m: {"result": int(m.group(1)) * int(m.group(2)), "expression": f"{m.group(1)}*{m.group(2)}"}, "math")
        self._add_rule(r"计算\s*(\d+)\s*/\s*(\d+)", 
                       lambda m: {"result": int(m.group(1)) // int(m.group(2)), "expression": f"{m.group(1)}/{m.group(2)}"}, "math")
        self._add_rule(r"搜索\s+(.+)", 
                       lambda m: {"query": m.group(1).strip(), "action": "search"}, "search")
        self._add_rule(r"查找\s+(.+)", 
                       lambda m: {"query": m.group(1).strip(), "action": "search"}, "search")
        
        self.category_templates = {
            "ai_concept": "人工智能(AI)是模拟人类智能的技术，包括机器学习、深度学习等领域。",
            "ml_concept": "机器学习是让计算机从数据中学习模式的技术，是AI的核心分支。",
            "dl_concept": "深度学习使用多层神经网络进行学习，是机器学习的高级形式。",
            "nlp_concept": "自然语言处理(NLP)让计算机理解和生成人类语言。"
        }
        
        entities = [
            ("人工智能", "concept", {"en": "AI", "definition": "模拟人类智能的技术"}),
            ("AI", "concept", {"cn": "人工智能"}),
            ("机器学习", "concept", {"en": "Machine Learning"}),
            ("深度学习", "concept", {"en": "Deep Learning"}),
            ("NLP", "concept", {"full": "自然语言处理"}),
            ("自然语言处理", "concept", {"abbr": "NLP"}),
            ("Python", "language", {}),
        ]
        for name, etype, attrs in entities:
            self.kg.add_entity(name, etype)
        
        relations = [
            ("深度学习", "机器学习", "is_subtype_of"),
            ("机器学习", "人工智能", "is_subtype_of"),
            ("机器学习", "AI", "is_subtype_of"),
            ("NLP", "AI", "is_application_of"),
            ("自然语言处理", "NLP", "same_as"),
        ]
        for src, tgt, rel in relations:
            self.kg.add_relation(src, tgt, rel)
    
    def _add_rule(self, pattern: str, handler: callable, category: str):
        self.rules.append({
            "pattern": re.compile(pattern, re.I),
            "handler": handler,
            "category": category
        })
    
    def process(self, text: str) -> Tuple[Dict, str, float]:
        self.stats["total_queries"] += 1
        
        for rule in self.rules:
            match = rule["pattern"].search(text)
            if match:
                self.stats["rule_hits"] += 1
                return rule["handler"](match), "rule", 1.0
        
        intent = self.parser.parse(text)
        if intent.action != ActionType.UNKNOWN:
            self.stats["pattern_hits"] += 1
            
            if intent.action == ActionType.QUESTION:
                text_lower = text.lower()
                if "人工智能" in text_lower or "ai" in text_lower:
                    return {"response": self.category_templates["ai_concept"]}, "template", 0.9
                elif "机器学习" in text_lower:
                    return {"response": self.category_templates["ml_concept"]}, "template", 0.9
                elif "深度学习" in text_lower:
                    return {"response": self.category_templates["dl_concept"]}, "template", 0.9
                elif "nlp" in text_lower or "自然语言" in text_lower:
                    return {"response": self.category_templates["nlp_concept"]}, "template", 0.9
            
            return {"intent": intent.action.value, "entities": intent.entities}, "symbolic", 0.8
        
        if text in self.learned_answers:
            self.stats["learned_hits"] += 1
            return {"response": self.learned_answers[text]}, "learned", 0.7
        
        for q, a in self.learned_answers.items():
            if self._similar(text, q) > 0.7:
                self.stats["learned_hits"] += 1
                return {"response": a, "similar_to": q}, "learned", 0.6
        
        self.stats["unknown_hits"] += 1
        return {"response": "我需要学习这个问题的答案", "need_learning": True}, "unknown", 0.0
    
    def _similar(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def learn(self, question: str, answer: str, confidence: float = 1.0):
        self.learned_answers[question] = answer
        self.rules.append({
            "pattern": re.compile(re.escape(question), re.I),
            "handler": lambda m: {"response": answer},
            "category": "learned"
        })
    
    def save(self, path: str):
        data = {
            "learned_answers": self.learned_answers,
            "rules_count": len(self.rules),
            "entities": list(self.kg.graph.nodes()),
            "relations": [(s, t, d.get("relation", "related")) 
                         for s, t, d in self.kg.graph.edges(data=True)],
            "stats": self.stats,
            "timestamp": time.time()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.learned_answers = data.get("learned_answers", {})
            self.stats = data.get("stats", self.stats)
        except FileNotFoundError:
            pass


class TeacherPool:
    def __init__(self):
        self.teachers = [
            {"name": "granite4:350m", "weight": 0.4, "speed": "fast"},
            {"name": "granite4:1b", "weight": 0.3, "speed": "medium"},
            {"name": "gemma3:1b", "weight": 0.2, "speed": "medium"},
            {"name": "granite4:3b", "weight": 0.1, "speed": "slow"},
        ]
        self.current_idx = 0
        self.call_counts = {t["name"]: 0 for t in self.teachers}
    
    def get_next_teacher(self) -> str:
        teacher = self.teachers[self.current_idx]["name"]
        self.current_idx = (self.current_idx + 1) % len(self.teachers)
        return teacher
    
    def query(self, model: str, prompt: str, timeout: int = 30) -> Optional[str]:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, 
                      "options": {"num_predict": 150, "temperature": 0.7}},
                timeout=timeout
            )
            if resp.status_code == 200:
                self.call_counts[model] += 1
                return resp.json().get("response", "")
        except Exception as e:
            pass
        return None
    
    def query_all(self, prompt: str) -> Dict[str, str]:
        results = {}
        for teacher in self.teachers[:3]:
            answer = self.query(teacher["name"], prompt)
            if answer:
                results[teacher["name"]] = answer
        return results
    
    def aggregate(self, responses: Dict[str, str]) -> Tuple[str, float]:
        if not responses:
            return "", 0.0
        
        weighted = []
        for teacher in self.teachers:
            name = teacher["name"]
            if name in responses:
                answer = responses[name]
                quality = min(1.0, len(answer) / 50)
                score = teacher["weight"] * quality
                weighted.append((answer, score))
        
        if not weighted:
            return list(responses.values())[0], 0.5
        
        weighted.sort(key=lambda x: -x[1])
        return weighted[0][0], weighted[0][1]


class Benchmark:
    def __init__(self):
        self.tests = [
            {"id": "tool_1", "q": "调用echo msg=hello", "category": "tool", 
             "check": lambda r: "message" in r and r["message"] == "hello"},
            {"id": "tool_2", "q": "计算2+2", "category": "math", 
             "check": lambda r: "result" in r and r["result"] == 4},
            {"id": "tool_3", "q": "计算10*5", "category": "math", 
             "check": lambda r: "result" in r and r["result"] == 50},
            {"id": "qa_1", "q": "什么是人工智能?", "category": "qa", 
             "check": lambda r: "人工智能" in str(r) or "AI" in str(r)},
            {"id": "qa_2", "q": "什么是机器学习?", "category": "qa", 
             "check": lambda r: "机器学习" in str(r) or "学习" in str(r)},
            {"id": "search_1", "q": "搜索人工智能", "category": "search", 
             "check": lambda r: ("query" in r and "人工智能" in r.get("query", "")) or 
                               ("action" in r and r.get("action") == "search") or
                               "search" in str(r).lower()},
        ]
        self.results = []
    
    def evaluate(self, student: FusionStudentModel) -> Dict:
        passed = 0
        results = []
        
        for test in self.tests:
            result, method, conf = student.process(test["q"])
            
            try:
                success = test["check"](result) and conf >= 0.7
            except Exception as e:
                print(f"    Check error for {test['id']}: {e}")
                success = False
            
            results.append({
                "id": test["id"],
                "category": test["category"],
                "success": success,
                "method": method,
                "confidence": conf
            })
            
            if success:
                passed += 1
        
        self.results = results
        return {
            "accuracy": passed / len(self.tests),
            "passed": passed,
            "total": len(self.tests),
            "details": results
        }


def main():
    print("="*60)
    print("Fusion Paradigm Model Training")
    print("="*60)
    
    output_dir = Path("fusion_paradigm/trained_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student = FusionStudentModel()
    student.bootstrap()
    print(f"\nStudent model initialized")
    print(f"  Rules: {len(student.rules)}")
    print(f"  Entities: {student.kg.graph.number_of_nodes()}")
    
    teachers = TeacherPool()
    benchmark = Benchmark()
    
    print("\nBaseline evaluation...")
    baseline = benchmark.evaluate(student)
    print(f"  Accuracy: {baseline['accuracy']:.1%}")
    
    test_questions = [
        ("什么是人工智能?", "qa"),
        ("什么是机器学习?", "category"),
        ("深度学习和机器学习的区别是什么?", "qa"),
        ("NLP是什么?", "qa"),
        ("Python是什么?", "qa"),
        ("调用echo msg=test", "tool"),
        ("计算5+3", "math"),
        ("搜索深度学习", "search"),
    ]
    
    print("\n" + "-"*60)
    print("Training iterations (Ctrl+C to stop)...")
    print("-"*60)
    
    max_iterations = 20
    target_accuracy = 0.85
    
    try:
        for iteration in range(1, max_iterations + 1):
            print(f"\n[Iteration {iteration}/{max_iterations}]")
            
            for question, category in test_questions:
                result, method, conf = student.process(question)
                
                if conf < 0.8:
                    print(f"  Learning: {question[:30]}...")
                    
                    teacher = teachers.get_next_teacher()
                    prompt = f"Answer concisely in Chinese: {question}"
                    answer = teachers.query(teacher, prompt)
                    
                    if answer:
                        student.learn(question, answer)
                        print(f"    Teacher: {teacher}, Learned!")
            
            if iteration % 5 == 0:
                results = benchmark.evaluate(student)
                print(f"\n  [Evaluation] Accuracy: {results['accuracy']:.1%}")
                
                if results["accuracy"] >= target_accuracy:
                    print(f"\n  Target accuracy {target_accuracy:.0%} reached!")
                    break
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    print("\n" + "-"*60)
    print("Final evaluation...")
    final_results = benchmark.evaluate(student)
    print(f"  Final accuracy: {final_results['accuracy']:.1%}")
    
    student.save(str(output_dir / "model.json"))
    
    with open(output_dir / "training_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "baseline_accuracy": baseline["accuracy"],
            "final_accuracy": final_results["accuracy"],
            "improvement": final_results["accuracy"] - baseline["accuracy"],
            "iterations": iteration,
            "teacher_calls": teachers.call_counts,
            "student_stats": student.stats,
            "test_results": final_results["details"]
        }, f, ensure_ascii=False, indent=2)
    
    print("\nFinal test:")
    print("-"*60)
    for test in benchmark.tests[:6]:
        result, method, conf = student.process(test["q"])
        status = "PASS" if test["check"](result) else "FAIL"
        print(f"  [{status}] {test['q']}")
        print(f"         Method: {method}, Conf: {conf:.2f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Baseline: {baseline['accuracy']:.1%} -> Final: {final_results['accuracy']:.1%}")
    print(f"Improvement: {(final_results['accuracy'] - baseline['accuracy']):.1%}")
    print("="*60)


if __name__ == "__main__":
    main()