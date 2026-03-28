# 修炼 - 轻量级符号推理引擎

非Transformer架构的AI系统，专注于工具调用和网络访问。

## 特性

| 指标 | 修炼 | Transformer |
|------|------|-------------|
| 复杂度 | O(n log n) | O(n²) |
| 内存 | <2GB | 14-350GB |
| 延迟 | 1-10ms | 500-2000ms |
| 参数 | <500M | 7B-175B |

## 安装

```bash
pip install xiulian
```

## 使用

```bash
# 交互模式
xiulian

# CLI
xiulian cli "搜索AI"
xiulian cli "调用echo msg=hello" --verbose

# GUI
xiulian gui

# Web
xiulian web --port 5000

# 性能测试
xiulian bench -n 1000
```

## Python API

```python
from xiulian import Engine

engine = Engine()
result = engine.process("调用echo msg=hello")
print(result.data)  # {'message': 'hello'}
```

## 架构

```
输入 → 符号解析(O(n)) → 意图分类
         ↓
      记忆检索(O(log n)) ← 知识库
         ↓
      图推理(O(V+E)) ← 关系图
         ↓
      工具执行/响应生成
```

核心组件：
- **SymbolicParser**: 模式匹配解析器
- **Memory**: Trie索引知识库
- **Graph**: 关系推理图
- **Tool**: 工具注册执行器
- **Web**: 网络访问模块

## 设计原理

基于第一性原理，融合历史AI范式：
1. **符号主义**: 规则匹配、可解释推理
2. **记忆网络**: 高效检索、增量更新
3. **图推理**: 关系遍历、路径缓存
4. **模板生成**: 结构化输出零计算

## 对比

与Transformer的本质区别：
- 知识表示：显式符号 vs 隐式参数
- 推理方式：规则+图遍历 vs 注意力机制
- 计算模式：稀疏检索 vs 稠密矩阵
- 可解释性：100% vs <10%
- 运营成本：极低 vs 很高

## License

GPLv3