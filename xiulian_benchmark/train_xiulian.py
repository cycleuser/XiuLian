#!/usr/bin/env python3
"""
训练修炼(XiuLian)非Transformer架构模型
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Engine, Memory, Graph, SymbolicParser

@dataclass
class TrainingConfig:
    """训练配置"""
    data_path: str = "data/training"
    output_path: str = "models/xiulian_trained"
    epochs: int = 10
    batch_size: int = 100
    learning_rate: float = 0.01
    
    # 模块权重
    parser_weight: float = 0.3
    memory_weight: float = 0.3
    graph_weight: float = 0.4

class XiuLianTrainer:
    """修炼模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.engine = Engine()
        self.training_stats = {
            "total_samples": 0,
            "parser_accuracy": 0.0,
            "memory_recall": 0.0,
            "graph_connectivity": 0.0
        }
    
    def load_training_data(self) -> List[Dict]:
        """加载训练数据"""
        data_file = Path(self.config.data_path) / "training_data.json"
        
        if not data_file.exists():
            print(f"⚠️  训练数据不存在: {data_file}")
            print("📝 生成默认训练数据...")
            return self._generate_default_data()
        
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_default_data(self) -> List[Dict]:
        """生成默认训练数据"""
        default_data = [
            # 工具调用
            {"input": "调用echo msg=hello", "action": "tool", "target": "echo", "params": {"msg": "hello"}},
            {"input": "使用time", "action": "tool", "target": "time", "params": {}},
            {"input": "执行calc expr=1+1", "action": "tool", "target": "calc", "params": {"expr": "1+1"}},
            
            # 搜索
            {"input": "搜索人工智能", "action": "search", "query": "人工智能"},
            {"input": "查找机器学习", "action": "search", "query": "机器学习"},
            {"input": "search AI", "action": "search", "query": "AI"},
            
            # 问答
            {"input": "什么是深度学习？", "action": "question", "topic": "深度学习"},
            {"input": "如何学习编程？", "action": "question", "topic": "学习编程"},
            {"input": "what is AI?", "action": "question", "topic": "AI"},
            
            # 网络访问
            {"input": "打开 https://example.com", "action": "web", "url": "https://example.com"},
            {"input": "访问 https://github.com", "action": "web", "url": "https://github.com"},
        ]
        
        # 保存默认数据
        Path(self.config.data_path).mkdir(parents=True, exist_ok=True)
        with open(Path(self.config.data_path) / "training_data.json", 'w', encoding='utf-8') as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已生成 {len(default_data)} 条默认训练数据")
        return default_data
    
    def train_parser(self, data: List[Dict]) -> float:
        """训练符号解析器"""
        print("\n📝 训练符号解析器...")
        
        correct = 0
        total = len(data)
        
        for item in data:
            intent = self.engine.parser.parse(item["input"])
            if intent.action == item["action"]:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        self.training_stats["parser_accuracy"] = accuracy
        
        print(f"   准确率: {accuracy:.2%}")
        return accuracy
    
    def train_memory(self, data: List[Dict]) -> float:
        """训练记忆模块"""
        print("\n🧠 训练记忆模块...")
        
        # 加载知识到记忆
        for item in data:
            if "query" in item:
                self.engine.memory.add(item["query"], item.get("answer", ""))
            elif "topic" in item:
                self.engine.memory.add(item["topic"], item.get("content", ""))
        
        # 测试检索
        recall_scores = []
        for item in data:
            if "query" in item:
                results = self.engine.memory.search(item["query"], top_k=1)
                if results:
                    recall_scores.append(results[0][2])  # 相似度分数
        
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        self.training_stats["memory_recall"] = avg_recall
        
        print(f"   召回率: {avg_recall:.2%}")
        return avg_recall
    
    def train_graph(self, data: List[Dict]) -> float:
        """训练图推理模块"""
        print("\n🕸️  训练图推理模块...")
        
        # 构建关系图
        for item in data:
            action = item["action"]
            self.engine.graph.add(action, node_type="action")
            
            if "target" in item:
                self.engine.graph.add(item["target"], node_type="entity")
                self.engine.graph.link(action, item["target"], relation="uses")
            
            if "query" in item:
                self.engine.graph.add(item["query"], node_type="query")
                self.engine.graph.link(action, item["query"], relation="searches")
        
        # 测试图连接性
        node_count = self.engine.graph.g.number_of_nodes()
        edge_count = self.engine.graph.g.number_of_edges()
        connectivity = (edge_count / node_count) if node_count > 0 else 0
        
        self.training_stats["graph_connectivity"] = min(connectivity, 1.0)
        
        print(f"   节点数: {node_count}")
        print(f"   边数: {edge_count}")
        print(f"   连接度: {connectivity:.2f}")
        return min(connectivity, 1.0)
    
    def save_model(self):
        """保存训练后的模型"""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        model_state = {
            "config": self.config.__dict__,
            "stats": self.training_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_dir / "model_state.json", 'w', encoding='utf-8') as f:
            json.dump(model_state, f, ensure_ascii=False, indent=2)
        
        # 保存记忆
        memory_data = {
            "store": self.engine.memory.store,
            "index": self.engine.memory.index
        }
        with open(output_dir / "memory.json", 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 模型已保存至: {output_dir}")
    
    def train(self):
        """完整训练流程"""
        print("="*60)
        print("🏋️  开始训练修炼(XiuLian)模型")
        print("="*60)
        
        # 加载数据
        data = self.load_training_data()
        self.training_stats["total_samples"] = len(data)
        print(f"\n📊 训练样本数: {len(data)}")
        
        # 训练各模块
        parser_acc = self.train_parser(data)
        memory_recall = self.train_memory(data)
        graph_conn = self.train_graph(data)
        
        # 计算综合得分
        overall_score = (
            parser_acc * self.config.parser_weight +
            memory_recall * self.config.memory_weight +
            graph_conn * self.config.graph_weight
        )
        
        print("\n" + "="*60)
        print("📊 训练结果")
        print("="*60)
        print(f"   解析器准确率: {parser_acc:.2%}")
        print(f"   记忆召回率: {memory_recall:.2%}")
        print(f"   图连接度: {graph_conn:.2f}")
        print(f"   综合得分: {overall_score:.2%}")
        
        # 保存模型
        self.save_model()
        
        return overall_score

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="训练修炼模型")
    parser.add_argument('--data', default='data/training', help='训练数据路径')
    parser.add_argument('--output', default='models/xiulian_trained', help='模型输出路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs
    )
    
    trainer = XiuLianTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()