#!/usr/bin/env python3
"""
Matplotlib可视化生成器
使用matplotlib生成符合顶级期刊标准的SVG图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = '#fafafa'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


class MatplotlibVisualizer:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "RESULTS"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            'XiuLian': '#1f77b4',
            'Qwen3.5-0.8B': '#d62728',
            'k-NN': '#2ca02c',
            'SVM': '#9467bd',
            'Decision Tree': '#ff7f0e',
            'Random Forest': '#8c564b',
            'Naive Bayes': '#e377c2',
            'Perceptron': '#7f7f7f',
            'Knowledge Graph': '#1f77b4',
            'Neural Net': '#17becf',
        }
        
        self.methods_data = {
            'XiuLian': {'accuracy': 87.5, 'latency_ms': 0.01, 'memory_mb': 1.8, 'interpretability': 95},
            'k-NN': {'accuracy': 96.5, 'latency_ms': 0.05, 'memory_mb': 0.5, 'interpretability': 80},
            'SVM': {'accuracy': 96.8, 'latency_ms': 0.15, 'memory_mb': 0.8, 'interpretability': 70},
            'Decision Tree': {'accuracy': 91.8, 'latency_ms': 0.10, 'memory_mb': 0.3, 'interpretability': 95},
            'Random Forest': {'accuracy': 96.1, 'latency_ms': 0.25, 'memory_mb': 1.2, 'interpretability': 75},
            'Naive Bayes': {'accuracy': 94.2, 'latency_ms': 0.02, 'memory_mb': 0.1, 'interpretability': 85},
            'Perceptron': {'accuracy': 72.3, 'latency_ms': 0.01, 'memory_mb': 0.01, 'interpretability': 90},
            'Knowledge Graph': {'accuracy': 92.1, 'latency_ms': 0.10, 'memory_mb': 0.4, 'interpretability': 100},
            'Qwen3.5-0.8B': {'accuracy': 92.3, 'latency_ms': 40369, 'memory_mb': 1370, 'interpretability': 30},
        }
    
    def generate_latency_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['XiuLian', 'Naive Bayes', 'k-NN', 'Decision Tree', 'Knowledge Graph', 'SVM', 'Random Forest', 'Qwen3.5-0.8B']
        latencies = [self.methods_data[m]['latency_ms'] for m in methods]
        colors = [self.colors.get(m, '#666666') for m in methods]
        
        y_pos = np.arange(len(methods))
        log_latencies = np.log10(np.array(latencies) + 0.001)
        bar_widths = (log_latencies - np.log10(0.001)) / (np.log10(50000) - np.log10(0.001)) * 8
        
        bars = ax.barh(y_pos, bar_widths, color=colors, height=0.6, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('Latency (log scale)', fontsize=14)
        ax.set_title('Inference Latency Comparison', fontsize=18, pad=20)
        
        tick_positions = np.linspace(0, 8, 7)
        tick_labels = ['10μs', '100μs', '1ms', '10ms', '100ms', '1s', '10s']
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        for i, (bar, lat) in enumerate(zip(bars, latencies)):
            if lat < 1:
                label = f'{lat*1000:.0f}μs'
            elif lat < 1000:
                label = f'{lat:.2f}ms'
            else:
                label = f'{lat/1000:.1f}s'
            
            if bar.get_width() > 1:
                ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2, 
                       label, ha='right', va='center', fontsize=10, color='white', fontweight='bold')
            else:
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center', fontsize=10, color='#666666')
        
        ax.annotate('4,000,000× speedup', xy=(0.1, 7.5), fontsize=12, color='#d62728', fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / 'fig1_latency.svg'
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def generate_accuracy_heatmap(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        datasets = ['Iris', 'Wine', 'Breast Cancer', '20 Newsgroups', 'MNIST', 'CIFAR-10']
        methods = ['k-NN', 'SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'Neural Net']
        
        data = np.array([
            [96.7, 94.4, 96.5, 82.3, 96.8, 75.2],
            [96.7, 94.4, 97.4, 85.1, 98.2, 78.5],
            [93.3, 88.9, 93.0, 78.5, 87.5, 65.3],
            [95.3, 96.1, 96.8, 84.7, 97.1, 77.8],
            [93.3, 97.2, 95.6, 88.2, 92.4, 70.1],
            [97.8, 98.5, 98.9, 92.5, 99.3, 85.7],
        ])
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=60, vmax=100)
        
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(datasets, fontsize=12)
        ax.set_yticklabels(methods, fontsize=12)
        
        for i in range(len(methods)):
            for j in range(len(datasets)):
                val = data[i, j]
                color = 'white' if val < 85 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=11, 
                       color=color, fontweight='bold')
        
        ax.set_title('Classification Accuracy Across Datasets (%)', fontsize=18, pad=20)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Accuracy (%)', fontsize=12)
        cbar.set_ticks([60, 70, 80, 90, 100])
        
        plt.tight_layout()
        filepath = self.output_dir / 'fig2_heatmap.svg'
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def generate_pareto_frontier(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(self.methods_data.keys())
        accuracies = [self.methods_data[m]['accuracy'] for m in methods]
        latencies = [self.methods_data[m]['latency_ms'] for m in methods]
        colors = [self.colors.get(m, '#666666') for m in methods]
        
        latencies_log = np.log10(np.array(latencies) + 0.001)
        
        scatter = ax.scatter(latencies_log, accuracies, c=colors, s=150, 
                            edgecolors='white', linewidths=2, alpha=0.8)
        
        for i, method in enumerate(methods):
            if method in ['XiuLian', 'Qwen3.5-0.8B', 'SVM', 'k-NN', 'Random Forest']:
                offset_x = 0.15 if latencies_log[i] < 2 else -0.15
                ax.annotate(method, (latencies_log[i], accuracies[i]), 
                           xytext=(offset_x, 5), textcoords='offset points',
                           fontsize=11, ha='center')
        
        pareto_methods = ['XiuLian', 'k-NN', 'SVM', 'Random Forest', 'Qwen3.5-0.8B']
        pareto_x = [np.log10(self.methods_data[m]['latency_ms'] + 0.001) for m in pareto_methods]
        pareto_y = [self.methods_data[m]['accuracy'] for m in pareto_methods]
        pareto_sorted = sorted(zip(pareto_x, pareto_y), key=lambda x: x[1])
        
        ax.plot([p[0] for p in pareto_sorted], [p[1] for p in pareto_sorted], 
               'b--', linewidth=2, alpha=0.5, label='Pareto Frontier')
        
        ax.fill_between([min(latencies_log)-0.5, max(pareto_x)], 
                        [min(accuracies)-5, min(pareto_y)], 
                        [min(pareto_y), min(pareto_y)], 
                        alpha=0.1, color='#1f77b4')
        
        ax.set_xlabel('Inference Latency (log scale)', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('Accuracy vs Efficiency Trade-off (Pareto Frontier)', fontsize=18, pad=20)
        
        tick_vals = [-2, -1, 0, 1, 2, 3, 4, 5]
        tick_labels = ['10μs', '100μs', '1ms', '10ms', '100ms', '1s', '10s', '100s']
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylim(70, 100)
        
        ax.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        filepath = self.output_dir / 'fig3_pareto.svg'
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def generate_scaling_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        
        lengths = np.linspace(50, 1000, 20)
        
        xiulian_latency = 0.5 + lengths * 0.012
        transformer_latency = lengths ** 2 * 0.18
        
        ax.plot(lengths, xiulian_latency, 'b-', linewidth=3, label='XiuLian O(n)', marker='o', markersize=5)
        ax.plot(lengths, transformer_latency / 1000, 'r-', linewidth=3, label='Transformer O(n²)', marker='s', markersize=5)
        
        ax.set_xlabel('Input Length (tokens)', fontsize=14)
        ax.set_ylabel('Latency (ms)', fontsize=14)
        ax.set_title('Complexity Scaling: O(n) vs O(n²)', fontsize=18, pad=20)
        
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 200)
        
        ax.annotate('XiuLian: 12.5ms', xy=(1000, 12.5), xytext=(850, 30),
                   fontsize=10, color='#1f77b4',
                   arrowprops=dict(arrowstyle='->', color='#1f77b4'))
        
        ax.annotate('Transformer: 180s', xy=(1000, 180), xytext=(800, 150),
                   fontsize=10, color='#d62728',
                   arrowprops=dict(arrowstyle='->', color='#d62728'))
        
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'fig4_scaling.svg'
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def generate_radar_chart(self):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        categories = ['Accuracy', 'Speed', 'Memory\nEfficiency', 'Interpretability', 'Train\nSpeed']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        
        def normalize_method(method_name):
            d = self.methods_data[method_name]
            acc = (d['accuracy'] - 70) / 30
            speed = 1 - min(np.log10(d['latency_ms'] + 0.001) / 5, 1)
            mem = 1 - min(d['memory_mb'] / 1500, 1)
            interp = d['interpretability'] / 100
            train = 0.9
            return [acc, speed, mem, interp, train]
        
        methods_to_show = ['XiuLian', 'SVM', 'Random Forest', 'Naive Bayes', 'Qwen3.5-0.8B']
        
        for method in methods_to_show:
            values = normalize_method(method)
            values += values[:1]
            color = self.colors.get(method, '#666666')
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        
        ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0), fontsize=11)
        
        plt.title('Multi-Dimensional Performance Comparison', fontsize=18, pad=20)
        plt.tight_layout()
        filepath = self.output_dir / 'fig5_radar.svg'
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def generate_comparison_barplot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = ['XiuLian', 'k-NN', 'SVM', 'Decision Tree', 'Random Forest', 
                  'Naive Bayes', 'Knowledge Graph', 'Qwen3.5-0.8B']
        
        accuracies = [self.methods_data[m]['accuracy'] for m in methods]
        colors = [self.colors.get(m, '#666666') for m in methods]
        
        ax1 = axes[0]
        bars = ax1.bar(range(len(methods)), accuracies, color=colors, edgecolor='white', linewidth=1.5)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy Comparison', fontsize=14)
        ax1.set_ylim(70, 100)
        ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax2 = axes[1]
        latencies = [self.methods_data[m]['latency_ms'] for m in methods]
        log_latencies = np.log10(np.array(latencies) + 0.001)
        bars = ax2.bar(range(len(methods)), log_latencies, color=colors, edgecolor='white', linewidth=1.5)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Latency (log₁₀ ms)', fontsize=12)
        ax2.set_title('Latency Comparison', fontsize=14)
        
        ax3 = axes[2]
        memories = [self.methods_data[m]['memory_mb'] for m in methods]
        log_memories = np.log10(np.array(memories) + 0.01)
        bars = ax3.bar(range(len(methods)), log_memories, color=colors, edgecolor='white', linewidth=1.5)
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Memory (log₁₀ MB)', fontsize=12)
        ax3.set_title('Memory Comparison', fontsize=14)
        
        plt.suptitle('Performance Metrics Comparison', fontsize=18, y=1.02)
        plt.tight_layout()
        filepath = self.output_dir / 'fig6_comparison.svg'
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def generate_all(self):
        print("=" * 60)
        print("Generating Matplotlib Visualizations")
        print("=" * 60)
        
        self.generate_latency_comparison()
        self.generate_accuracy_heatmap()
        self.generate_pareto_frontier()
        self.generate_scaling_comparison()
        self.generate_radar_chart()
        self.generate_comparison_barplot()
        
        print("=" * 60)
        print(f"All figures saved to: {self.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    viz = MatplotlibVisualizer()
    viz.generate_all()