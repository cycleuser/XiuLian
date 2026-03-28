# XiuLian (修炼)

A lightweight non-Transformer agent orchestration engine for tool calling and intelligent agent workflows.

## Features

- **Symbolic Parser**: O(n) complexity intent recognition without neural networks
- **Tool Registry**: Dynamic tool registration with parallel execution support
- **Knowledge Graph**: Entity-relationship reasoning with path finding
- **Workflow Engine**: Declarative workflow orchestration with conditional branching
- **Memory System**: Trie-indexed knowledge storage for fast retrieval
- **Web Client**: Async HTTP client for API integrations

## Installation

```bash
pip install xiulian
```

Or from source:

```bash
git clone https://github.com/cycleuser/XiuLian.git
cd XiuLian
pip install -e .
```

## Quick Start

```python
from xiulian import Agent, ToolRegistry, ToolDef

# Create agent
agent = Agent()

# Process natural language commands
result = agent.process("调用echo msg=Hello World")
print(result.data)  # {'message': 'Hello World'}

# Register custom tools
def my_tool(x: int, y: int) -> dict:
    return {"sum": x + y}

agent.tools.register(ToolDef(
    name="add",
    func=my_tool,
    description="Add two numbers",
    params_schema={"x": "int", "y": "int"}
))

result = agent.process("调用add x=5 y=3")
print(result.data)  # {'sum': 8}
```

## CLI Usage

```bash
# Interactive mode
python -m xiulian.cli

# Single command
python -m xiulian.cli "调用time"

# Benchmark
python -m xiulian.cli --benchmark
```

## Performance

Compared to TinyLlama (1.1B parameters):

| Metric | XiuLian | TinyLlama | Improvement |
|--------|---------|-----------|-------------|
| Latency | 0.082ms | 650ms | 7927x faster |
| Memory | 69.1MB | 2200MB | 32x efficient |
| Throughput | 19,458/s | 1.5/s | 12,972x higher |
| Tool Accuracy | 100% | 87% | +13% |

## Architecture

```
User Input → Parser (Intent Recognition)
         ↓
    Memory (Trie Index) ←→ Knowledge Graph
         ↓
    Tool Registry (Execution)
         ↓
    Workflow Engine (Orchestration)
         ↓
    Result (Response)
```

## API

### Agent

Main orchestration class:

```python
agent = Agent()
agent.process(text: str) -> Result
agent.benchmark(iterations: int = 100) -> dict
```

### Parser

Symbolic intent parser:

```python
parser = Parser()
intent = parser.parse(text: str)  # Returns Intent object
```

### ToolRegistry

Tool management:

```python
registry = ToolRegistry()
registry.register(tool: ToolDef)
registry.execute(name: str, **params) -> Result
registry.execute_parallel(calls: list) -> list[Result]
```

### Memory

Knowledge storage with Trie indexing:

```python
memory = Memory()
memory.add(key: str, value: Any)
memory.search(query: str, top_k: int = 5) -> list[tuple]
```

### KnowledgeGraph

Entity-relationship reasoning:

```python
kg = KnowledgeGraph()
kg.add_entity(name: str, entity_type: str = "entity")
kg.add_relation(source: str, target: str, relation: str)
kg.find_path(source: str, target: str) -> list[str]
```

### WorkflowEngine

Declarative workflow orchestration:

```python
workflow = Workflow(steps=[
    WorkflowStep(id="step1", action="tool", params={"name": "echo"}),
    WorkflowStep(id="step2", action="tool", params={"name": "calc"}, condition="step1.success")
])
engine = WorkflowEngine(tools=registry)
result = engine.run(workflow)
```

## Examples

See `examples/demo_agent.py` for a complete demonstration.

## Development

```bash
# Run tests
python -m pytest xiulian/tests/ -v

# Run benchmark
python scripts/run_benchmark.py

# Generate figures
python scripts/generate_figures.py
```

## Academic Paper

See `article.md` (English) and `article_cn.md` (Chinese) for the full academic paper comparing XiuLian with Transformer-based approaches.

## License

GPL-3.0 License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{xiulian2024,
  title={XiuLian: A Non-Transformer Architecture for Efficient Agent Orchestration},
  author={XiuLian Team},
  year={2024},
  url={https://github.com/cycleuser/XiuLian}
}
```