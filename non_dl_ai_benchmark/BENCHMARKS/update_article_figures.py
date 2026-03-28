#!/usr/bin/env python3
"""
更新Article.md中的SVG图表为matplotlib生成的版本
"""

import re
from pathlib import Path

def update_figures():
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "RESULTS"
    article_path = base_dir.parent / "Article.md"
    
    with open(article_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    figure_mappings = [
        ('Fig. 1', 'fig1_latency.svg', 'Inference latency comparison (log scale).'),
        ('Fig. 2', 'fig2_heatmap.svg', 'Classification accuracy heatmap across benchmark datasets.'),
        ('Fig. 3', 'fig3_pareto.svg', 'Accuracy vs efficiency trade-off (Pareto Frontier).'),
        ('Fig. 4', 'fig4_scaling.svg', 'Complexity scaling comparison.'),
        ('Fig. 5', 'fig5_radar.svg', 'Multi-dimensional performance comparison.'),
    ]
    
    for fig_num, svg_file, description in figure_mappings:
        svg_path = results_dir / svg_file
        if not svg_path.exists():
            print(f"Warning: {svg_path} not found")
            continue
        
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        pattern = rf'\*\*{re.escape(fig_num)}\. [^*]+\*\*[^\n]*\n\n<svg[^>]*>.*?</svg>'
        
        new_figure = f'**{fig_num}. {description}**\n\n{svg_content}'
        
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content[:match.start()] + new_figure + content[match.end():]
            print(f"Updated {fig_num}")
        else:
            print(f"Could not find {fig_num} in Article.md")
    
    with open(article_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nUpdated {article_path}")

if __name__ == "__main__":
    update_figures()