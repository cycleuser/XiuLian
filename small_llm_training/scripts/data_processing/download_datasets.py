#!/usr/bin/env python3
"""
数据集下载脚本（中国大陆优化版）
使用国内镜像源和数据集
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def download_small_datasets():
    """下载小型数据集（国内镜像优化）"""
    
    print("📦 开始下载训练数据集（使用国内镜像）...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # 1. 尝试从 ModelScope 下载数据集
    print("\n1️⃣  尝试从 ModelScope 下载数据集...")
    try:
        from modelscope.msdatasets import MsDataset
        
        # 下载通用文本数据
        print("   下载通用文本数据...")
        try:
            dataset = MsDataset.load('AI-ModelScope/wikipedia-cn-20230720-filtered', subset_name='default', split='train')
            # 取前10000条
            data_list = []
            for i, item in enumerate(tqdm(dataset, desc="处理数据")):
                if i >= 10000:
                    break
                data_list.append({'text': item['text'], 'title': item.get('title', '')})
            
            # 保存
            save_path = DATA_DIR / "wikipedia_small"
            from datasets import Dataset
            wiki_dataset = Dataset.from_list(data_list)
            wiki_dataset.save_to_disk(str(save_path))
            print(f"   ✅ Wikipedia 中文数据: {len(data_list)} 条")
        except Exception as e:
            print(f"   ⚠️  ModelScope Wikipedia 下载失败: {e}")
            print("   📥 使用备选方案...")
            create_fallback_data()
            
    except ImportError:
        print("   ⚠️  ModelScope 未安装，使用 HuggingFace 镜像...")
        download_from_huggingface_mirror()
    
    # 2. 创建合成数据（确保始终有数据可用）
    print("\n2️⃣  创建高质量合成数据...")
    create_synthetic_data()
    
    # 3. 下载 Alpaca 中文版（如果有）
    print("\n3️⃣  下载中文指令数据...")
    try:
        # 使用 HuggingFace 镜像
        alpaca = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:5000]")
        alpaca.save_to_disk(str(DATA_DIR / "alpaca_chinese"))
        print(f"   ✅ Alpaca 中文: {len(alpaca)} 条")
    except Exception as e:
        print(f"   ⚠️  Alpaca 中文下载失败: {e}")
        print("   将使用合成数据")
    
    print("\n✅ 数据下载完成！")
    print(f"📁 数据保存在: {DATA_DIR}")

def download_from_huggingface_mirror():
    """从 HuggingFace 镜像下载数据"""
    
    print("   使用 HuggingFace 镜像...")
    
    # Wikipedia
    try:
        wiki = load_dataset("wikipedia", "20220301.simple", split="train[:5000]")
        wiki.save_to_disk(str(DATA_DIR / "wikipedia_small"))
        print(f"   ✅ Wikipedia: {len(wiki)} 条")
    except Exception as e:
        print(f"   ⚠️  Wikipedia 下载失败: {e}")
    
    # Alpaca
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train[:5000]")
        alpaca.save_to_disk(str(DATA_DIR / "alpaca"))
        print(f"   ✅ Alpaca: {len(alpaca)} 条")
    except Exception as e:
        print(f"   ⚠️  Alpaca 下载失败: {e}")

def create_fallback_data():
    """创建备选数据（网络不可用时）"""
    
    print("   📝 创建备选文本数据...")
    
    texts = [
        "人工智能是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。",
        "深度学习是机器学习的一个分支，它基于人工神经网络的研究，试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。",
        "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。",
        "计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉。",
    ]
    
    # 扩展数据
    expanded_texts = []
    for text in texts:
        # 创建变体
        expanded_texts.append({"text": text, "source": "fallback"})
        expanded_texts.append({"text": f"什么是{text[:20]}？{text}", "source": "qa"})
        expanded_texts.append({"text": f"简介：{text}", "source": "summary"})
    
    save_path = DATA_DIR / "fallback_data"
    from datasets import Dataset
    dataset = Dataset.from_list(expanded_texts)
    dataset.save_to_disk(str(save_path))
    print(f"   ✅ 备选数据: {len(expanded_texts)} 条")

def create_synthetic_data():
    """创建高质量合成数据（确保训练可用）"""
    
    synthetic_data = []
    
    # AI 相关问答
    qa_pairs = [
        ("什么是人工智能？", "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统，包括学习、推理、问题解决、感知和语言理解等能力。"),
        ("机器学习和深度学习有什么区别？", "机器学习是AI的子集，使用算法从数据中学习模式和做出决策。深度学习是机器学习的子集，使用多层神经网络来模拟人脑的学习过程，特别擅长处理复杂的非结构化数据。"),
        ("什么是自然语言处理？", "自然语言处理（NLP）是AI的一个领域，专注于计算机与人类语言之间的交互。它包括文本分析、情感分析、机器翻译、问答系统等多个应用领域。"),
        ("神经网络的工作原理是什么？", "神经网络是一种模仿人脑结构的计算系统，由多层节点组成。每个节点接收输入，通过权重计算，激活函数处理，然后传递给下一层。通过反向传播算法调整权重来学习模式。"),
        ("什么是过拟合？如何避免？", "过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象。可以通过增加训练数据、使用正则化、Dropout、早停等方法来避免。"),
        ("解释一下梯度下降算法。", "梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数相对于参数的梯度，沿着梯度的反方向更新参数，逐步找到损失函数的最小值。"),
        ("什么是迁移学习？", "迁移学习是一种机器学习技术，将在一个任务上训练好的模型应用到另一个相关任务上。它可以大大减少训练时间和所需的数据量。"),
        ("卷积神经网络主要用于什么？", "卷积神经网络（CNN）主要用于图像处理任务，如图像分类、目标检测、图像分割等。它通过卷积层自动提取图像的层次特征。"),
        ("什么是强化学习？", "强化学习是一种机器学习方法，智能体通过与环境交互，根据奖励信号学习最优策略。常用于游戏、机器人控制、自动驾驶等领域。"),
        ("Transformer架构的优势是什么？", "Transformer架构使用自注意力机制，可以并行处理序列数据，解决了RNN的长期依赖问题。它是现代大语言模型的基础架构。"),
    ]
    
    # 数学问题
    math_problems = [
        ("计算 15 + 27", "15 + 27 = 42"),
        ("计算 100 - 37", "100 - 37 = 63"),
        ("计算 8 × 9", "8 × 9 = 72"),
        ("计算 144 ÷ 12", "144 ÷ 12 = 12"),
        ("计算 2^10", "2^10 = 1024"),
        ("求 12 和 18 的最大公约数", "12 和 18 的最大公约数是 6"),
        ("计算圆的面积，半径为 5", "圆的面积 = πr² = π × 5² = 25π ≈ 78.54"),
    ]
    
    # 代码示例
    code_examples = [
        ("用Python写一个Hello World程序", "print('Hello, World!')"),
        ("写一个计算阶乘的递归函数", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"),
        ("写一个列表反转的代码", "my_list = [1, 2, 3, 4, 5]\nreversed_list = my_list[::-1]"),
        ("写一个判断素数的函数", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    ]
    
    # 文本生成
    text_generation = [
        "人工智能技术的发展正在深刻改变我们的生活方式。从智能手机到自动驾驶汽车，AI技术已经渗透到我们生活的方方面面。",
        "在医疗领域，人工智能可以帮助医生进行疾病诊断，分析医学影像，甚至预测疾病的发展趋势。",
        "教育领域也在积极应用人工智能技术。个性化学习系统可以根据学生的学习进度和特点，提供定制化的学习内容。",
        "环境保护是人工智能应用的另一个重要领域。AI可以帮助监测空气质量，预测天气变化，优化能源使用。",
        "未来，随着技术的不断进步，人工智能将在更多领域发挥重要作用，为人类社会带来更多便利和可能性。",
    ]
    
    # 组合数据
    synthetic_data.extend([
        {"instruction": q, "output": a, "type": "qa", "category": "ai"}
        for q, a in qa_pairs
    ])
    
    synthetic_data.extend([
        {"instruction": q, "output": a, "type": "math", "category": "math"}
        for q, a in math_problems
    ])
    
    synthetic_data.extend([
        {"instruction": q, "output": a, "type": "code", "category": "programming"}
        for q, a in code_examples
    ])
    
    synthetic_data.extend([
        {"text": text, "type": "article", "category": "general"}
        for text in text_generation
    ])
    
    # 保存
    synthetic_path = DATA_DIR / "synthetic.json"
    with open(synthetic_path, 'w', encoding='utf-8') as f:
        json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 合成数据: {len(synthetic_data)} 条")
    print(f"   📁 保存至: {synthetic_path}")
    
    # 同时保存为 JSONL 格式
    synthetic_jsonl = DATA_DIR / "synthetic.jsonl"
    with open(synthetic_jsonl, 'w', encoding='utf-8') as f:
        for item in synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("\n" + "="*60)
    print("      超小语言模型训练 - 数据下载（中国大陆优化版）")
    print("="*60 + "\n")
    
    try:
        download_small_datasets()
        
        print("\n" + "="*60)
        print("✅ 数据集下载完成！")
        print("="*60)
        
        # 显示数据集统计
        print("\n📊 数据集统计:")
        for dataset_dir in DATA_DIR.iterdir():
            if dataset_dir.is_dir():
                print(f"   - {dataset_dir.name}")
        
        if (DATA_DIR / "synthetic.json").exists():
            print("\n💡 提示: 即使网络问题，合成数据也能确保训练可以进行")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 建议: 使用合成数据进行训练")
        sys.exit(0)  # 不退出，合成数据已创建

if __name__ == "__main__":
    main()