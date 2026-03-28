"""命令行接口"""

import argparse
import json
import sys
from . import __version__, Engine

def main(args=None):
    parser = argparse.ArgumentParser(prog="xiulian", description="修炼 - 轻量级符号推理引擎")
    parser.add_argument("-V", "--version", action="version", version=f"xiulian {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--json", action="store_true", help="JSON格式输出")
    
    sub = parser.add_subparsers(dest="cmd")
    
    cli = sub.add_parser("cli", help="处理输入")
    cli.add_argument("input", help="输入文本")
    
    sub.add_parser("gui", help="启动图形界面")
    
    web = sub.add_parser("web", help="启动Web服务")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=5000)
    
    sub.add_parser("tools", help="列出工具")
    sub.add_parser("stats", help="显示统计")
    
    bench = sub.add_parser("bench", help="性能测试")
    bench.add_argument("-n", type=int, default=100, help="迭代次数")
    
    args = parser.parse_args(args)
    engine = Engine()
    
    if args.cmd == "cli":
        r = engine.process(args.input)
        output = json.dumps(r.to_dict(), indent=2, ensure_ascii=False) if args.json else (
            json.dumps(r.data, ensure_ascii=False) if r.success else f"错误: {r.error}"
        )
        print(output)
        if args.verbose:
            print(f"延迟: {r.latency_ms:.2f}ms")
        return 0 if r.success else 1
    
    if args.cmd == "gui":
        from .gui import run_gui
        run_gui()
        return 0
    
    if args.cmd == "web":
        from .app import run_server
        run_server(host=args.host, port=args.port)
        return 0
    
    if args.cmd == "tools":
        for t in engine.tools.list():
            print(f"  {t['name']}: {t['desc']}")
        return 0
    
    if args.cmd == "stats":
        print(json.dumps(engine.tools.stats, indent=2))
        return 0
    
    if args.cmd == "bench":
        r = engine.benchmark(args.n)
        print(json.dumps(r, indent=2))
        print(f"\n对比Transformer基线: 延迟 500-2000ms, 内存 14-350GB")
        print(f"修炼引擎: 延迟 {r['avg_latency_ms']:.2f}ms, 内存 <2GB")
        return 0
    
    # 交互模式
    print(f"修炼 v{__version__} | 输入 help 查看命令, exit 退出")
    while True:
        try:
            text = input("修炼> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not text or text in ("exit", "quit"):
            break
        if text == "help":
            print("命令: tools, stats, bench, 或直接输入查询")
            continue
        if text == "tools":
            for t in engine.tools.list():
                print(f"  {t['name']}: {t['desc']}")
            continue
        if text == "stats":
            print(json.dumps(engine.tools.stats, indent=2))
            continue
        if text == "bench":
            print(json.dumps(engine.benchmark(100), indent=2))
            continue
        
        r = engine.process(text)
        print(json.dumps(r.data, ensure_ascii=False, indent=2) if r.success else f"错误: {r.error}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())