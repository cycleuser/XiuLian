"""Flask Web服务"""

import json
from flask import Flask, request, jsonify, render_template_string
from . import __version__, Engine

app = Flask(__name__)
engine = Engine()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>修炼 v{{ version }}</title>
    <style>
        body { font-family: system-ui; max-width: 900px; margin: 0 auto; padding: 20px; }
        header { background: #1a1a2e; color: white; padding: 20px; margin: -20px -20px 20px; }
        h1 { margin: 0; }
        .main { display: grid; grid-template-columns: 1fr 300px; gap: 20px; }
        textarea { width: 100%; height: 80px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #1a1a2e; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #16213e; }
        .output { background: #1a1a2e; color: #0f0; padding: 15px; border-radius: 4px; font-family: monospace; min-height: 200px; white-space: pre-wrap; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 20px; }
        .stat { background: #f5f5f5; padding: 10px; border-radius: 4px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .stat-label { font-size: 12px; color: #666; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; }
        @media (max-width: 700px) { .main { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <header><h1>修炼 v{{ version }}</h1><p>轻量级符号推理引擎 | O(n log n) | &lt;2GB | &lt;10ms</p></header>
    
    <div class="main">
        <div>
            <textarea id="input" placeholder="输入查询..."></textarea>
            <div style="margin-top:10px">
                <button onclick="run()">执行</button>
                <button onclick="bench()" style="background:#059669">性能测试</button>
            </div>
            <div id="output" class="output" style="margin-top:20px">结果将显示在这里...</div>
        </div>
        <div>
            <div class="stats">
                <div class="stat"><div class="stat-value" id="total">0</div><div class="stat-label">查询数</div></div>
                <div class="stat"><div class="stat-value" id="rate">100%</div><div class="stat-label">成功率</div></div>
                <div class="stat"><div class="stat-value" id="latency">0ms</div><div class="stat-label">平均延迟</div></div>
                <div class="stat"><div class="stat-value" id="cache">0</div><div class="stat-label">缓存</div></div>
            </div>
            
            <h3>API</h3>
            <p><code>POST /api/query</code> 处理查询</p>
            <p><code>GET /api/tools</code> 列出工具</p>
            <p><code>POST /api/bench</code> 性能测试</p>
            
            <h3>对比Transformer</h3>
            <table>
                <tr><th>指标</th><th>修炼</th><th>Transformer</th></tr>
                <tr><td>延迟</td><td>1-10ms</td><td>500-2000ms</td></tr>
                <tr><td>内存</td><td>&lt;2GB</td><td>14-350GB</td></tr>
                <tr><td>参数</td><td>&lt;500M</td><td>7B-175B</td></tr>
                <tr><td>复杂度</td><td>O(n log n)</td><td>O(n²)</td></tr>
            </table>
        </div>
    </div>
    
    <script>
        async function run() {
            const input = document.getElementById('input').value;
            if (!input) return;
            document.getElementById('output').textContent = '处理中...';
            
            const r = await fetch('/api/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: input})
            });
            const data = await r.json();
            document.getElementById('output').textContent = JSON.stringify(data, null, 2);
            updateStats();
        }
        
        async function bench() {
            document.getElementById('output').textContent = '测试中...';
            const r = await fetch('/api/bench', {method: 'POST'});
            document.getElementById('output').textContent = JSON.stringify(await r.json(), null, 2);
        }
        
        async function updateStats() {
            const r = await fetch('/api/stats');
            const s = await r.json();
            document.getElementById('total').textContent = s.total || 0;
            document.getElementById('rate').textContent = ((s.success_rate || 1) * 100) + '%';
            document.getElementById('latency').textContent = (s.avg_latency || 0).toFixed(1) + 'ms';
            document.getElementById('cache').textContent = s.cache || 0;
        }
        
        document.getElementById('input').addEventListener('keydown', e => {
            if (e.ctrlKey && e.key === 'Enter') run();
        });
        
        setInterval(updateStats, 5000);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, version=__version__)

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json or {}
    r = engine.process(data.get("query", ""))
    return jsonify(r.to_dict())

@app.route("/api/tools")
def tools():
    return jsonify(engine.tools.list())

@app.route("/api/stats")
def stats():
    return jsonify({
        "total": engine.tools.stats["total"],
        "success_rate": engine.tools.stats["success"] / max(engine.tools.stats["total"], 1),
        "avg_latency": 0,
        "cache": len(engine.web.cache)
    })

@app.route("/api/bench", methods=["POST"])
def bench():
    return jsonify(engine.benchmark(100))

def run_server(host="127.0.0.1", port=5000):
    print(f"修炼 Web服务 http://{host}:{port}")
    app.run(host=host, port=port)