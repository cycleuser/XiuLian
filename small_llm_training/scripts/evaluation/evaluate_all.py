#!/usr/bin/env python3
"""
模型评估脚本
评估训练好的模型
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: Path, device: str = 'cpu'):
        print(f"\n📥 加载模型: {model_path.name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # 设置设备
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        # 模型信息
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   参数量: {self.total_params:,}")
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """计算困惑度"""
        total_loss = 0
        total_length = 0
        
        for text in tqdm(texts, desc="计算困惑度"):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
            
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_length += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_length
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_qa(self, qa_pairs: List[Dict]) -> Dict[str, float]:
        """评估问答能力"""
        correct = 0
        total = len(qa_pairs)
        
        for qa in tqdm(qa_pairs, desc="评估问答"):
            question = qa['question']
            expected_keywords = qa.get('keywords', [])
            
            generated = self.generate(question, max_length=128)
            
            # 简单的关键词匹配
            if any(keyword.lower() in generated.lower() for keyword in expected_keywords):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy}
    
    def evaluate_generation_quality(self, prompts: List[str]) -> Dict[str, Any]:
        """评估生成质量"""
        generations = []
        
        for prompt in tqdm(prompts, desc="生成文本"):
            generated = self.generate(prompt, max_length=200)
            generations.append({
                'prompt': prompt,
                'generated': generated,
                'length': len(generated)
            })
        
        # 统计
        avg_length = sum(g['length'] for g in generations) / len(generations)
        
        return {
            'num_generations': len(generations),
            'avg_length': avg_length,
            'generations': generations[:5]  # 保存前5个示例
        }

def load_test_data() -> Dict[str, Any]:
    """加载测试数据"""
    test_file = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    
    if not test_file.exists():
        # 使用合成测试数据
        return {
            'texts': [
                "人工智能是计算机科学的一个分支。",
                "机器学习使用算法从数据中学习。",
                "自然语言处理专注于计算机与人类语言的交互。"
            ],
            'qa_pairs': [
                {
                    'question': "什么是人工智能？",
                    'keywords': ["计算机科学", "智能", "AI"]
                },
                {
                    'question': "机器学习和深度学习有什么区别？",
                    'keywords': ["机器学习", "深度学习", "神经网络"]
                },
                {
                    'question': "解释一下过拟合。",
                    'keywords': ["训练", "新数据", "泛化"]
                }
            ],
            'generation_prompts': [
                "人工智能的未来发展方向是",
                "深度学习在医疗领域的应用包括",
                "自然语言处理的主要挑战是"
            ]
        }
    
    # 加载真实测试数据
    texts = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item['text'])
    
    return {
        'texts': texts[:100],  # 使用前100条
        'qa_pairs': [],  # 需要构建
        'generation_prompts': texts[:10]
    }

def evaluate_all_models():
    """评估所有训练好的模型"""
    
    print("\n" + "="*60)
    print("      评估所有模型")
    print("="*60 + "\n")
    
    # 查找所有训练好的模型
    final_dir = PROJECT_ROOT / "models" / "final"
    pretrained_dir = PROJECT_ROOT / "models" / "pretrained"
    
    models = []
    
    # 检查训练后的模型
    if final_dir.exists():
        for model_dir in final_dir.iterdir():
            if model_dir.is_dir():
                models.append(('trained', model_dir))
    
    # 检查预训练模型
    if pretrained_dir.exists():
        for model_dir in pretrained_dir.iterdir():
            if model_dir.is_dir():
                models.append(('pretrained', model_dir))
    
    if not models:
        print("❌ 没有找到可评估的模型")
        return
    
    # 加载测试数据
    print("📂 加载测试数据...")
    test_data = load_test_data()
    print(f"   文本数量: {len(test_data['texts'])}")
    
    # 评估结果
    results = {}
    
    for model_type, model_path in models:
        model_name = model_path.name
        
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name} ({model_type})")
        print(f"{'='*60}")
        
        try:
            evaluator = ModelEvaluator(model_path)
            
            # 困惑度
            print("\n📊 计算困惑度...")
            perplexity = evaluator.calculate_perplexity(test_data['texts'][:50])
            print(f"   困惑度: {perplexity:.2f}")
            
            # 生成质量
            print("\n📊 评估生成质量...")
            gen_quality = evaluator.evaluate_generation_quality(
                test_data['generation_prompts'][:5]
            )
            print(f"   平均生成长度: {gen_quality['avg_length']:.1f}")
            
            # 问答能力
            if test_data['qa_pairs']:
                print("\n📊 评估问答能力...")
                qa_result = evaluator.evaluate_qa(test_data['qa_pairs'])
                print(f"   准确率: {qa_result['accuracy']:.2%}")
            else:
                qa_result = {'accuracy': 0}
            
            # 保存结果
            results[model_name] = {
                'type': model_type,
                'params': evaluator.total_params,
                'perplexity': perplexity,
                'generation_quality': gen_quality,
                'qa_accuracy': qa_result['accuracy']
            }
            
            print(f"\n✅ {model_name} 评估完成")
            
        except Exception as e:
            print(f"\n❌ {model_name} 评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)
    print(f"📁 结果保存在: {results_file}")
    
    # 显示对比
    print("\n📊 模型对比:")
    print(f"{'模型':<20} {'参数量':>15} {'困惑度':>10} {'问答准确率':>12}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<20} {result['params']:>15,} {result['perplexity']:>10.2f} {result['qa_accuracy']:>11.2%}")

def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument('--model', type=str, help='指定模型路径')
    args = parser.parse_args()
    
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ 模型不存在: {model_path}")
            return
        
        test_data = load_test_data()
        evaluator = ModelEvaluator(model_path)
        
        perplexity = evaluator.calculate_perplexity(test_data['texts'][:50])
        print(f"\n困惑度: {perplexity:.2f}")
    else:
        evaluate_all_models()

if __name__ == "__main__":
    main()