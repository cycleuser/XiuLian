#!/usr/bin/env python3
"""
修炼智能体编排引擎示例

展示如何使用修炼进行：
1. 工具调用
2. 知识管理
3. 工作流编排
4. 并行执行
"""

import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from xiulian import Agent, Result, ToolDef, Workflow, WorkflowStep


def demo_basic_tools():
    """基础工具调用演示"""
    print("=" * 60)
    print("1. 基础工具调用")
    print("=" * 60)
    
    agent = Agent()
    
    test_cases = [
        "调用echo msg=你好世界",
        "调用calc expr=2+2*3",
        "调用time",
        "调用random min=1 max=100",
        "调用tools",
    ]
    
    for case in test_cases:
        result = agent.process(case)
        print(f"\n输入: {case}")
        print(f"结果: {result.data if result.success else f'错误: {result.error}'}")
        print(f"延迟: {result.latency_ms:.3f}ms")


def demo_custom_tool():
    """自定义工具演示"""
    print("\n" + "=" * 60)
    print("2. 自定义工具")
    print("=" * 60)
    
    agent = Agent()
    
    def weather_tool(**kwargs):
        city = kwargs.get('city', '北京')
        return {
            "city": city,
            "temperature": "22°C",
            "weather": "晴",
            "humidity": "45%",
        }
    
    def translate_tool(**kwargs):
        text = kwargs.get('text', '')
        target = kwargs.get('target', 'en')
        return {
            "original": text,
            "translated": f"[翻译结果: {text} -> {target}]",
            "target_lang": target,
        }
    
    agent.register_tool(
        name="weather",
        func=weather_tool,
        description="查询天气",
        params_schema={"city": {"type": "string", "description": "城市名称"}}
    )
    
    agent.register_tool(
        name="translate",
        func=translate_tool,
        description="翻译文本",
        params_schema={
            "text": {"type": "string", "description": "要翻译的文本"},
            "target": {"type": "string", "description": "目标语言", "default": "en"}
        }
    )
    
    test_cases = [
        "调用weather city=上海",
        "调用translate text=你好 target=en",
        "调用tools",
    ]
    
    for case in test_cases:
        result = agent.process(case)
        print(f"\n输入: {case}")
        print(f"结果: {result.data}")


def demo_knowledge():
    """知识管理演示"""
    print("\n" + "=" * 60)
    print("3. 知识管理")
    print("=" * 60)
    
    agent = Agent()
    
    agent.add_knowledge("人工智能", {
        "definition": "模拟人类智能的机器系统",
        "fields": ["机器学习", "深度学习", "自然语言处理"]
    })
    agent.add_knowledge("机器学习", {
        "definition": "从数据中学习的AI子集",
        "types": ["监督学习", "无监督学习", "强化学习"]
    })
    agent.add_knowledge("深度学习", {
        "definition": "使用多层神经网络的机器学习方法",
        "architectures": ["CNN", "RNN", "Transformer"]
    })
    
    agent.add_entity("人工智能", "concept")
    agent.add_entity("机器学习", "concept")
    agent.add_entity("深度学习", "concept")
    agent.add_entity("Transformer", "architecture")
    
    agent.add_relation("人工智能", "机器学习", "包含")
    agent.add_relation("机器学习", "深度学习", "包含")
    agent.add_relation("深度学习", "Transformer", "使用")
    
    test_cases = [
        "什么是人工智能?",
        "搜索深度学习",
        "机器学习是什么?",
    ]
    
    for case in test_cases:
        result = agent.process(case)
        print(f"\n输入: {case}")
        print(f"结果: {result.data}")
    
    path = agent.knowledge.find_path("人工智能", "Transformer")
    print(f"\n推理路径: {' -> '.join(path)}")


def demo_workflow():
    """工作流编排演示"""
    print("\n" + "=" * 60)
    print("4. 工作流编排")
    print("=" * 60)
    
    agent = Agent()
    
    def step1_validate(**kwargs):
        data = kwargs.get('data', '')
        return {"valid": len(data) > 0, "data": data}
    
    def step2_process(**kwargs):
        data = kwargs.get('data', '')
        return {"processed": data.upper(), "length": len(data)}
    
    def step3_format(**kwargs):
        processed = kwargs.get('processed', '')
        return {"result": f"处理完成: {processed}"}
    
    agent.register_tool("validate", step1_validate, "验证数据")
    agent.register_tool("process_data", step2_process, "处理数据")
    agent.register_tool("format_result", step3_format, "格式化结果")
    
    agent.register_workflow(
        name="data_pipeline",
        description="数据处理流水线",
        steps=[
            {"id": "validate", "action": "validate", "params": {"data": "{input.data}"}},
            {"id": "process", "action": "process_data", "params": {"data": "{results.validate.data}"}},
            {"id": "format", "action": "format_result", "params": {"processed": "{results.process.processed}"}},
        ]
    )
    
    result = agent.workflows.execute("data_pipeline", {"data": "hello world"})
    print(f"工作流执行结果: {result.data}")


def demo_parallel():
    """并行执行演示"""
    print("\n" + "=" * 60)
    print("5. 并行执行")
    print("=" * 60)
    
    agent = Agent()
    
    calls = [
        ("echo", {"msg": "任务1"}),
        ("echo", {"msg": "任务2"}),
        ("echo", {"msg": "任务3"}),
        ("calc", {"expr": "1+1"}),
        ("time", {}),
    ]
    
    start = time.time()
    results = agent.tools.execute_parallel(calls)
    elapsed = time.time() - start
    
    print(f"并行执行 {len(calls)} 个任务:")
    for i, r in enumerate(results):
        print(f"  任务{i+1}: {r.data}")
    print(f"总耗时: {elapsed*1000:.2f}ms")


async def demo_async():
    """异步执行演示"""
    print("\n" + "=" * 60)
    print("6. 异步执行")
    print("=" * 60)
    
    agent = Agent()
    
    texts = ["调用echo msg=异步1", "调用time", "调用calc expr=3*4"]
    
    start = time.time()
    results = await agent.parallel_process(texts)
    elapsed = time.time() - start
    
    print(f"异步处理 {len(texts)} 个请求:")
    for text, result in zip(texts, results):
        print(f"  {text}: {result.data}")
    print(f"总耗时: {elapsed*1000:.2f}ms")


def demo_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("7. 性能基准测试")
    print("=" * 60)
    
    agent = Agent()
    
    print("运行1000次迭代...")
    result = agent.benchmark(1000)
    
    print(f"成功率: {result['success_rate']:.1%}")
    print(f"平均延迟: {result['avg_latency_ms']:.3f}ms")
    print(f"最小延迟: {result['min_latency_ms']:.3f}ms")
    print(f"最大延迟: {result['max_latency_ms']:.3f}ms")
    print(f"吞吐量: {result['throughput_per_s']:.0f} 请求/秒")


def demo_interactive():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("8. 交互模式 (输入 'quit' 退出)")
    print("=" * 60)
    
    agent = Agent()
    
    while True:
        try:
            text = input("\n修炼> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        
        if not text or text.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        
        if text.lower() == "help":
            print("""
可用命令:
  调用<工具名> [参数]  - 执行工具
  搜索<关键词>          - 搜索知识
  什么是<概念>?         - 提问
  tools                 - 列出所有工具
  stats                 - 显示统计
  quit                  - 退出
            """)
            continue
        
        if text.lower() == "tools":
            for t in agent.tools.list_tools():
                print(f"  {t['name']}: {t['description']}")
            continue
        
        if text.lower() == "stats":
            stats = agent.get_stats()
            print(f"工具调用: {stats['tools']['total_calls']}")
            print(f"知识条目: {stats['knowledge_items']}")
            continue
        
        result = agent.process(text)
        if result.success:
            print(f"结果: {result.data}")
        else:
            print(f"错误: {result.error}")
        print(f"延迟: {result.latency_ms:.3f}ms")


def main():
    print("修炼智能体编排引擎演示")
    print("=" * 60)
    
    demo_basic_tools()
    demo_custom_tool()
    demo_knowledge()
    demo_workflow()
    demo_parallel()
    asyncio.run(demo_async())
    demo_benchmark()
    
    print("\n" + "=" * 60)
    print("演示完成!")


if __name__ == "__main__":
    main()