"""
简化版融合训练脚本
直接运行训练循环，简化异常处理
"""

import asyncio
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_paradigm.fusion_training import (
    FusionStudentModel, OllamaTeacherPool, TrainingLoop
)
from fusion_paradigm.evaluation import StandardBenchmark


async def simple_training():
    print("="*60)
    print("融合范式模型训练")
    print("="*60)
    
    output_dir = Path("fusion_paradigm/trained_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] 初始化...")
    student = FusionStudentModel()
    student.bootstrap_from_xiulian()
    print("  学生模型初始化完成")
    
    teacher_pool = OllamaTeacherPool()
    benchmark = StandardBenchmark()
    print("  教师池和基准测试准备完成")
    
    print("\n[2/4] 基线评估...")
    results = benchmark.evaluate_student(student)
    baseline_score = benchmark.get_overall_score()
    print(f"  基线得分: {baseline_score:.2%}")
    
    baseline_report = benchmark.generate_report()
    with open(output_dir / "baseline_report.md", 'w', encoding='utf-8') as f:
        f.write(baseline_report)
    print("  基线报告已保存")
    
    print("\n[3/4] 开始训练循环...")
    print("  目标: 100次迭代或85%准确率")
    print("-"*60)
    
    max_iterations = 100
    target_accuracy = 0.85
    
    training_loop = TrainingLoop(student, teacher_pool, str(output_dir))
    
    for iteration in range(1, max_iterations + 1):
        try:
            progress = await training_loop.iteration_step()
            
            print(f"\n迭代 {iteration}/{max_iterations}")
            print(f"  教师: {progress.current_teacher}")
            print(f"  准确率: {progress.accuracy:.2%}")
            print(f"  损失: {progress.loss:.4f}")
            print(f"  样本数: {progress.samples_generated}")
            
            if iteration % 10 == 0:
                results = benchmark.evaluate_student(student)
                current_score = benchmark.get_overall_score()
                print(f"  [评估] 当前得分: {current_score:.2%}")
                
                if current_score >= target_accuracy:
                    print(f"\n[TARGET] 达到目标准确率 {target_accuracy:.2%}!")
                    break
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"  [ERROR] 迭代错误: {e}")
            await asyncio.sleep(2)
    
    print("\n[4/4] 保存结果...")
    
    student.save(str(output_dir))
    print("  学生模型已保存")
    
    final_results = benchmark.evaluate_student(student)
    final_score = benchmark.get_overall_score()
    
    final_report = benchmark.generate_report()
    with open(output_dir / "final_report.md", 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    benchmark.save_results(str(output_dir / "final_results.json"))
    
    await teacher_pool.close()
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"  基线得分: {baseline_score:.2%}")
    print(f"  最终得分: {final_score:.2%}")
    print(f"  提升幅度: {(final_score - baseline_score):.2%}")
    print("="*60)
    
    return final_score


if __name__ == "__main__":
    asyncio.run(simple_training())