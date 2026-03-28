# 🧠 非深度学习AI策略综合研究项目

## 研究目标

从第一性原理实现并对比测试所有非Transformer、非深度学习的AI策略。

## 📚 AI范式分类

### 1. 符号主义 (Symbolism)

#### 1.1 专家系统 (Expert Systems)
- **起源**: Feigenbaum et al. (1970s)
- **核心**: 知识库 + 推理引擎
- **复杂度**: O(n) 规则匹配

#### 1.2 产生式系统 (Production Systems)
- **起源**: Newell & Simon (1972)
- **核心**: IF-THEN规则 + 冲突解决
- **代表**: OPS5, CLIPS

#### 1.3 案例推理 (Case-Based Reasoning)
- **起源**: Schank (1982)
- **核心**: 相似度匹配 + 案例检索
- **复杂度**: O(n) ~ O(n log n)

### 2. 连接主义 - 非深度学习

#### 2.1 感知机 (Perceptron)
- **起源**: Rosenblatt (1958)
- **核心**: 线性分类器
- **论文**: Rosenblatt, F. (1958). "The perceptron: A probabilistic model..."

#### 2.2 Hopfield网络
- **起源**: Hopfield (1982)
- **核心**: 联想记忆、能量函数
- **论文**: Hopfield, J.J. (1982). "Neural networks and physical systems..."

#### 2.3 Boltzmann机
- **起源**: Hinton & Sejnowski (1983)
- **核心**: 随机神经网络
- **论文**: Hinton, G.E. & Sejnowski, T.J. (1983)

### 3. 进化计算 (Evolutionary Computation)

#### 3.1 遗传算法 (Genetic Algorithm)
- **起源**: Holland (1975)
- **核心**: 选择、交叉、变异
- **论文**: Holland, J.H. (1975). "Adaptation in Natural and Artificial Systems"

#### 3.2 遗传编程 (Genetic Programming)
- **起源**: Koza (1992)
- **核心**: 程序树进化
- **论文**: Koza, J.R. (1992). "Genetic Programming"

#### 3.3 演化策略 (Evolution Strategies)
- **起源**: Rechenberg (1973)
- **核心**: 实数优化
- **论文**: Rechenberg, I. (1973). "Evolutionsstrategie"

### 4. 模糊逻辑 (Fuzzy Logic)

#### 4.1 模糊推理系统
- **起源**: Zadeh (1965)
- **核心**: 隶属度函数 + 模糊规则
- **论文**: Zadeh, L.A. (1965). "Fuzzy sets"

### 5. 贝叶斯方法

#### 5.1 朴素贝叶斯
- **核心**: 贝叶斯定理 + 特征独立假设
- **论文**: Maron, M.E. (1961). "Automatic indexing..."

#### 5.2 贝叶斯网络
- **起源**: Pearl (1985)
- **核心**: 概率图模型 + 条件独立
- **论文**: Pearl, J. (1985). "Bayesian networks..."

### 6. 决策树方法

#### 6.1 ID3/C4.5
- **起源**: Quinlan (1986)
- **核心**: 信息增益 + 递归分裂
- **论文**: Quinlan, J.R. (1986). "Induction of decision trees"

#### 6.2 随机森林
- **起源**: Ho (1995), Breiman (2001)
- **核心**: 集成学习 + 特征随机
- **论文**: Breiman, L. (2001). "Random Forests"

### 7. 核方法

#### 7.1 支持向量机 (SVM)
- **起源**: Vapnik (1995)
- **核心**: 最大间隔 + 核技巧
- **论文**: Cortes, C. & Vapnik, V. (1995). "Support-vector networks"

### 8. 实例学习

#### 8.1 k-近邻 (k-NN)
- **起源**: Fix & Hodges (1951)
- **核心**: 距离度量 + 投票
- **论文**: Fix, E. & Hodges, J.L. (1951)

### 9. 强化学习 - 非深度

#### 9.1 Q-Learning
- **起源**: Watkins (1989)
- **核心**: 值函数迭代
- **论文**: Watkins, C.J.C.H. (1989). "Learning from delayed rewards"

### 10. 图推理

#### 10.1 语义网络
- **起源**: Quillian (1966)
- **核心**: 节点 + 边 + 推理

#### 10.2 知识图谱推理
- **起源**: Google (2012)
- **核心**: 实体关系 + 路径推理

## 📊 对比维度

### 性能指标
1. **时间复杂度**: O(?) 
2. **空间复杂度**: O(?)
3. **训练时间**: ms/s/min
4. **推理延迟**: ms
5. **内存占用**: MB/GB

### 效果指标
1. **准确率**: %
2. **召回率**: %
3. **F1分数**: %
4. **泛化能力**: 训练/测试差距

### 工程指标
1. **可解释性**: %
2. **可扩展性**: 并行度
3. **鲁棒性**: 噪声容忍度
4. **数据效率**: 样本效率

## 📁 项目结构

```
non_dl_ai_benchmark/
├── README.md                   # 本文件
├── REFERENCES.md              # 完整参考文献 ⭐
├── METHODS/                   # 各方法实现
│   ├── 01_symbolic/           # 符号主义
│   │   ├── expert_system.py
│   │   ├── production_system.py
│   │   └── case_based_reasoning.py
│   ├── 02_connectionist/      # 连接主义
│   │   ├── perceptron.py
│   │   ├── hopfield.py
│   │   └── boltzmann.py
│   ├── 03_evolutionary/       # 进化计算
│   │   ├── genetic_algorithm.py
│   │   ├── genetic_programming.py
│   │   └── evolution_strategy.py
│   ├── 04_fuzzy/              # 模糊逻辑
│   │   └── fuzzy_inference.py
│   ├── 05_bayesian/           # 贝叶斯
│   │   ├── naive_bayes.py
│   │   └── bayesian_network.py
│   ├── 06_decision_tree/      # 决策树
│   │   ├── id3_c45.py
│   │   └── random_forest.py
│   ├── 07_kernel/             # 核方法
│   │   └── svm.py
│   ├── 08_instance/           # 实例学习
│   │   └── knn.py
│   ├── 09_reinforcement/      # 强化学习
│   │   └── q_learning.py
│   └── 10_graph/              # 图推理
│       ├── semantic_network.py
│       └── knowledge_graph.py
├── BENCHMARKS/                # 基准测试
│   ├── benchmark_suite.py     # 测试套件
│   ├── datasets/              # 测试数据集
│   └── metrics/               # 评估指标
├── COMPARISON/                # 对比分析
│   ├── compare_all.py         # 全面对比
│   ├── visualize.py           # 可视化
│   └── statistical_test.py    # 统计检验
├── REPORTS/                   # 研究报告
│   ├── methodology.md         # 方法论
│   ├── results.md             # 实验结果
│   ├── analysis.md            # 分析讨论
│   └── conclusion.md          # 结论
├── scripts/
│   ├── run_all.sh             # 一键运行
│   ├── train_all.sh           # 训练所有方法
│   └── compare.sh             # 对比分析
└── requirements.txt
```

## 🚀 快速开始

```bash
# 一键运行所有实验
./scripts/run_all.sh

# 训练所有方法
./scripts/train_all.sh

# 生成对比报告
./scripts/compare.sh
```

## 📚 主要参考文献

### 经典著作
1. Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
2. Mitchell, T.M. (1997). *Machine Learning*
3. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*

### 关键论文
见 [REFERENCES.md](REFERENCES.md) 获取完整列表

## 🔬 实验设计

### 任务类型
1. **分类任务**: 文本分类、图像分类
2. **回归任务**: 数值预测
3. **聚类任务**: 无监督学习
4. **推理任务**: 逻辑推理、知识推理
5. **优化任务**: 函数优化、组合优化
6. **决策任务**: 序列决策、博弈

### 数据集
1. **分类**: Iris, Wine, 20NewsGroups
2. **推理**: CLUTRR, bAbI
3. **优化**: 标准测试函数
4. **决策**: GridWorld, OpenAI Gym

## 📊 预期成果

1. **方法库**: 所有AI范式的第一性原理实现
2. **基准测试**: 标准化评估框架
3. **对比分析**: 全面对比报告
4. **最佳实践**: 各方法适用场景指南
5. **学术贡献**: 研究论文

## 📝 许可证

GPLv3