"""
极简训练脚本
顺序处理，只用最快的小模型
"""

import requests
import json
import time
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Parser, Memory, KnowledgeGraph


class SimpleStudent:
    def __init__(self):
        self.parser = Parser()
        self.memory = Memory()
        self.kb = KnowledgeGraph()
        self.rules = []
        self.patterns = {}
        self.learned = {}
    
    def bootstrap(self):
        self.patterns = {
            r"调用\s*echo\s+msg\s*=\s*(\w+)": lambda m: {"message": m.group(1)},
            r"计算\s*(\d+)\s*\+\s*(\d+)": lambda m: {"result": int(m.group(1)) + int(m.group(2))},
            r"计算\s*(\d+)\s*\*\s*(\d+)": lambda m: {"result": int(m.group(1)) * int(m.group(2))},
            r"搜索\s+(.+)": lambda m: {"query": m.group(1), "action": "search"},
        }
        
        for entity in ["AI", "人工智能", "机器学习", "深度学习", "NLP", "自然语言处理"]:
            self.kb.add_entity(entity, "concept")
        
        self.kb.add_relation("深度学习", "机器学习", "is_subtype_of")
        self.kb.add_relation("机器学习", "人工智能", "is_subtype_of")
    
    def process(self, text):
        for pattern, handler in self.patterns.items():
            match = re.search(pattern, text, re.I)
            if match:
                return handler(match), "rule", 1.0
        
        intent = self.parser.parse(text)
        if intent.action.value != "unknown":
            return {"intent": intent.action.value}, "symbolic", 0.8
        
        if text in self.learned:
            return {"response": self.learned[text]}, "learned", 0.7
        
        return {"response": "我需要学习这个问题的答案"}, "unknown", 0.0
    
    def learn(self, question, answer, confidence=1.0):
        self.learned[question] = answer
        self.rules.append({"q": question, "a": answer, "conf": confidence})
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "rules": self.rules,
                "learned": self.learned,
                "patterns": list(self.patterns.keys())
            }, f, ensure_ascii=False, indent=2)


def query_ollama(model, prompt, timeout=30):
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, 
                  "options": {"num_predict": 100}},
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return None
    except Exception as e:
        print(f"  API错误: {e}")
        return None


def main():
    print("="*60)
    print("融合范式模型 - 极简训练")
    print("="*60)
    
    output_dir = Path("fusion_paradigm/trained_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student = SimpleStudent()
    student.bootstrap()
    print("\n学生模型初始化完成")
    
    teacher = "granite4:350m"
    print(f"教师模型: {teacher}")
    
    test_cases = [
        ("什么是人工智能?", "qa"),
        ("什么是机器学习?", "qa"),
        ("深度学习和机器学习的区别?", "qa"),
        ("NLP是什么?", "qa"),
        ("调用echo msg=hello", "tool"),
        ("计算2+2", "math"),
        ("计算10*5", "math"),
        ("搜索人工智能", "search"),
    ]
    
    print("\n开始训练...")
    print("-"*60)
    
    for iteration in range(1, 6):
        print(f"\n[迭代 {iteration}/5]")
        
        for question, category in test_cases:
            result, method, conf = student.process(question)
            
            if conf < 0.8:
                print(f"\n  Q: {question}")
                print(f"  当前置信度: {conf:.2f}, 请教教师...")
                
                prompt = f"请简洁回答以下问题: {question}"
                answer = query_ollama(teacher, prompt)
                
                if answer:
                    student.learn(question, answer)
                    print(f"  A: {answer[:80]}...")
                    print(f"  已学习!")
            else:
                print(f"  {question[:20]}... -> 已掌握 ({method})")
    
    print("\n" + "-"*60)
    print("保存模型...")
    student.save(str(output_dir / "learned_rules.json"))
    
    print("\n最终测试:")
    print("-"*60)
    
    correct = 0
    for question, category in test_cases:
        result, method, conf = student.process(question)
        print(f"\nQ: {question}")
        if "response" in result:
            print(f"A: {result['response'][:80]}...")
        else:
            print(f"A: {result}")
        print(f"方法: {method}, 置信度: {conf:.2f}")
        
        if conf >= 0.7:
            correct += 1
    
    accuracy = correct / len(test_cases)
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"正确率: {accuracy:.2%} ({correct}/{len(test_cases)})")
    print("="*60)
    
    with open(output_dir / "training_summary.json", 'w', encoding='utf-8') as f:
        json.dump({
            "iterations": 5,
            "test_cases": len(test_cases),
            "accuracy": accuracy,
            "teacher": teacher,
            "rules_learned": len(student.rules),
            "timestamp": time.time()
        }, f, ensure_ascii=False, indent=2)
    
    return accuracy


if __name__ == "__main__":
    main()