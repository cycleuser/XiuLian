#!/usr/bin/env python3
"""
修炼智能体编排引擎完整测试
"""

import sys
import time
import json
import tracemalloc
import statistics
from datetime import datetime

sys.path.insert(0, '/Users/fred/Documents/GitHub/cycleuser/XiuLian')

from xiulian import Agent

TOOL_CALLING_CASES = [
    ("调用echo msg=hello", "echo", True),
    ("调用calc expr=2+2", "calc", True),
    ("调用time", "time", True),
    ("调用random min=1 max=100", "random", True),
    ("计算 100*5", "calc", True),
    ("calc 1+2+3", "calc", True),
]

QUESTION_CASES = [
    ("什么是人工智能?", "人工智能", True),
    ("机器学习是什么?", "机器学习", True),
    ("深度学习是什么?", "深度学习", True),
]

INTENT_CASES = [
    ("搜索机器学习教程", "search"),
    ("查找Python文档", "search"),
    ("调用echo工具", "tool"),
    ("使用calc计算", "tool"),
    ("什么是AI?", "question"),
    ("如何学习编程?", "question"),
]

def measure_performance(agent, test_cases, iterations=100):
    latencies = []
    correct = 0
    total = 0
    
    for _ in range(iterations):
        for case in test_cases:
            start = time.perf_counter()
            result = agent.process(case[0])
            intent = agent.parser.parse(case[0])
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            total += 1
            
            if len(case) > 2 and case[2]:
                if result.success:
                    if intent.action.value == "tool":
                        tool_name = intent.entities.get("target", "").lower()
                        if tool_name == case[1].lower():
                            correct += 1
                    elif intent.action.value in ("question", "search"):
                        query = case[1]
                        if query in agent.memory.store:
                            correct += 1
                    else:
                        correct += 1
            elif len(case) > 1:
                if isinstance(case[1], str):
                    if intent.action.value == case[1]:
                        correct += 1
    
    latencies.sort()
    n = len(latencies)
    
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "std_ms": statistics.stdev(latencies) if n > 1 else 0,
    }

def measure_memory():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_full_benchmark():
    print("=" * 60)
    print("XiuLian 智能体编排引擎完整测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tracemalloc.start()
    agent = Agent()
    
    agent.add_knowledge("人工智能", {"definition": "模拟人类智能的机器系统"})
    agent.add_knowledge("机器学习", {"definition": "从数据中学习的AI子集"})
    agent.add_knowledge("深度学习", {"definition": "使用多层神经网络"})
    
    results = {}
    
    print("\n1. 工具调用测试...")
    results["tool_calling"] = measure_performance(agent, TOOL_CALLING_CASES, 50)
    print(f"   准确率: {results['tool_calling']['accuracy']:.1%}")
    print(f"   平均延迟: {results['tool_calling']['avg_latency_ms']:.3f}ms")
    
    print("\n2. 问题问答测试...")
    results["question_answering"] = measure_performance(agent, QUESTION_CASES, 50)
    print(f"   准确率: {results['question_answering']['accuracy']:.1%}")
    print(f"   平均延迟: {results['question_answering']['avg_latency_ms']:.3f}ms")
    
    print("\n3. 意图识别测试...")
    results["intent_recognition"] = measure_performance(agent, INTENT_CASES, 50)
    print(f"   准确率: {results['intent_recognition']['accuracy']:.1%}")
    print(f"   平均延迟: {results['intent_recognition']['avg_latency_ms']:.3f}ms")
    
    memory_mb = measure_memory()
    
    print("\n4. 吞吐量测试...")
    start = time.time()
    count = 0
    for _ in range(100):
        for case in TOOL_CALLING_CASES[:5]:
            agent.process(case[0])
            count += 1
    elapsed = time.time() - start
    throughput = count / elapsed
    print(f"   吞吐量: {throughput:.0f} 请求/秒")
    
    print("\n5. 输入长度复杂度测试...")
    complexity_results = []
    test_inputs = [
        "测试" * 10,
        "测试" * 50,
        "测试" * 100,
        "测试" * 200,
        "测试" * 500,
    ]
    
    for inp in test_inputs:
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            agent.process(inp)
            latencies.append((time.perf_counter() - start) * 1000)
        avg_latency = statistics.mean(latencies)
        complexity_results.append({
            "length": len(inp),
            "latency_ms": avg_latency
        })
        print(f"   长度 {len(inp):4d}: {avg_latency:.3f}ms")
    
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "platform": sys.platform,
            "python_version": sys.version,
        },
        "benchmarks": results,
        "memory_mb": memory_mb,
        "throughput_per_sec": throughput,
        "complexity_test": complexity_results,
        "summary": {
            "total_samples": sum(r["samples"] for r in results.values()),
            "overall_accuracy": sum(r["correct"] for r in results.values()) / sum(r["samples"] for r in results.values()),
            "avg_latency_ms": statistics.mean([r["avg_latency_ms"] for r in results.values()]),
            "p95_latency_ms": max(r["p95_ms"] for r in results.values()),
        }
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总样本数: {final_results['summary']['total_samples']}")
    print(f"总体准确率: {final_results['summary']['overall_accuracy']:.1%}")
    print(f"平均延迟: {final_results['summary']['avg_latency_ms']:.3f}ms")
    print(f"P95延迟: {final_results['summary']['p95_latency_ms']:.3f}ms")
    print(f"内存占用: {memory_mb:.1f}MB")
    print(f"吞吐量: {throughput:.0f} 请求/秒")
    
    with open('/Users/fred/Documents/GitHub/cycleuser/XiuLian/xiulian/data/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 xiulian/data/benchmark_results.json")
    
    return final_results

if __name__ == "__main__":
    run_full_benchmark()