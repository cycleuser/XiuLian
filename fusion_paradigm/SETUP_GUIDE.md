# 融合范式训练系统 - 环境设置与运行指南

## 当前状态

训练框架已完成设计和实现，但需要Python环境才能运行。

**已完成组件**:
- ✅ 多教师知识蒸馏框架
- ✅ FusionStudentModel学生模型架构
- ✅ TeacherPool教师模型池（支持ollama）
- ✅ 符号知识库（继承XiuLian）
- ✅ 神经网络模块（轻量MLP）
- ✅ 迭代训练循环
- ✅ 标准化测试评估框架
- ✅ 持续迭代与进度记录

**文件清单**:
```
fusion_paradigm/
├── ARCHITECTURE.md          # 架构设计文档
├── fusion_engine.py         # 原推理引擎（备用）
├── fusion_training.py       # 核心训练框架
├── evaluation.py            # 标准化测试评估
├── run_training.py          # 主运行脚本
└── SETUP_GUIDE.md           # 本文档
```

## 环境要求

### 必需组件
1. **Python 3.8+** - 主运行环境
2. **ollama** - 教师模型提供者（已安装）
3. **基础依赖**:
   - aiohttp (异步HTTP客户端)
   - numpy (可选，神经网络模块)

### 可选组件
- GPU支持（训练会更快，但CPU也可运行）

## 安装步骤

### 方式1: 安装Python（推荐）

1. **下载Python**:
   - 访问 https://www.python.org/downloads/
   - 下载Python 3.10+ Windows安装包

2. **安装Python**:
   ```
   - 运行安装包
   - 勾选"Add Python to PATH"
   - 完成安装
   ```

3. **验证安装**:
   ```bash
   python --version
   pip --version
   ```

4. **安装依赖**:
   ```bash
   pip install aiohttp numpy
   ```

### 方式2: 使用Anaconda

1. **下载Anaconda**:
   - https://www.anaconda.com/download

2. **创建环境**:
   ```bash
   conda create -n fusion python=3.10
   conda activate fusion
   pip install aiohttp numpy
   ```

## 运行训练

### 快速启动

```bash
cd C:/Users/frede/Documents/GitHub/XiuLian
python fusion_paradigm/run_training.py
```

### 训练流程

训练会自动执行：
1. **初始化** - 加载符号知识库，连接教师模型池
2. **基线评估** - 测试初始性能
3. **训练循环** - 多教师轮流训练学生模型
4. **定期评估** - 每10次迭代评估一次
5. **进度报告** - 每5次迭代生成报告
6. **最终保存** - 保存训练后的模型和结果

### 可停止训练

训练过程中可以按 `Ctrl+C` 停止：
- 会自动保存当前进度
- 下次可以从保存的状态继续

### 查看结果

训练完成后查看：
```
fusion_paradigm/trained_model/
├── baseline_report.md       # 基线评估报告
├── progress_report.md       # 进度报告
├── final_report.md          # 最终评估报告
├── symbolic_kb.json         # 学到的符号规则
├── neural_module.json       # 神经网络参数
├── session_state.json       # 会话状态
└── evaluation_results.json  # 评估数据
```

## 训练目标

### 性能目标
- 总体得分 ≥ 85% (优秀)
- 通过率 ≥ 70%
- 各分类得分平衡

### 测试类别
| 类别 | 权重 | 说明 |
|------|------|------|
| 工具调用 | 25% | 准确识别和执行工具 |
| 知识问答 | 30% | 回答AI/ML相关问题 |
| 搜索推理 | 15% | 搜索信息处理 |
| 数学计算 | 15% | 基础数学运算 |
| 语义理解 | 15% | 语义分析能力 |

## 教师模型

系统使用以下ollama模型作为教师：

| 模型 | 参数量 | 权重 | 特长 |
|------|--------|------|------|
| qwen3:14b | 14.8B | 35% | 推理能力强 |
| gemma3:4b | 4.3B | 25% | QA能力好 |
| granite4:3b | 3.4B | 20% | 代码能力强 |
| granite4:1b | 1.6B | 15% | 响应速度快 |
| granite4:350m | 352M | 5% | 轻量辅助 |

## 技术架构

### 融合范式设计

```
问题输入 → [符号快速筛选] → [神经网络补充] → [融合决策] → 输出

符号路径:
- 规则匹配（确定性，O(n)）
- 知识图谱推理（结构化）
- 模式识别（快速）

神经网络路径:
- 软模式学习（模糊推理）
- 教师输出学习
- 语义关联

融合层:
- 符号优先（高置信度）
- 网络补充（低置信度）
- 投票机制（多路径）
```

### 第一性原理

AI = 模式识别 + 推理计算 + 决策输出

- **符号范式**: 确定性、高效率、可解释
- **神经范式**: 泛化性、模糊推理、学习能力
- **融合范式**: 结合两者优势

## 下一步

**安装Python后**:
1. 验证ollama运行: `ollama list`
2. 安装依赖: `pip install aiohttp`
3. 运行训练: `python fusion_paradigm/run_training.py`
4. 查看报告: `fusion_paradigm/trained_model/`

**持续迭代**:
- 系统会自动持续训练
- 定期保存进度
- 达到目标后停止或用户中断

---

**等待用户安装Python以启动训练...**