"""
融合范式训练主程序
整合所有组件，实现持续迭代训练
"""

import asyncio
import sys
import signal
import json
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_paradigm.fusion_training import (
    FusionStudentModel, OllamaTeacherPool, TrainingLoop, TrainingProgress
)
from fusion_paradigm.evaluation import (
    StandardBenchmark, ContinuousEvaluation, EvaluationResult
)


class FusionTrainingSession:
    """融合训练会话
    
    管理完整的训练生命周期：
    1. 初始化组件
    2. 基线评估
    3. 训练循环
    4. 定期评估
    5. 进度报告
    6. 异常处理
    """
    
    def __init__(self, output_dir: str = "fusion_paradigm/trained_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.student = FusionStudentModel()
        self.teacher_pool = OllamaTeacherPool()
        self.benchmark = StandardBenchmark()
        self.evaluator = ContinuousEvaluation(self.student, self.benchmark)
        self.training_loop = TrainingLoop(
            self.student, self.teacher_pool, str(self.output_dir)
        )
        
        self.running = True
        self.iteration_count = 0
        self.max_iterations = 100
        self.eval_interval = 10
        self.report_interval = 5
        
        self.session_log: List[Dict] = []
        self.start_time = time.time()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\n\n收到停止信号，正在保存进度...")
        self.running = False
        self._save_session_state()
        print("进度已保存，训练停止。")
    
    async def initialize(self):
        print("="*70)
        print("融合范式模型训练系统")
        print("="*70)
        
        print("\n[阶段1] 初始化组件...")
        
        self.student.bootstrap_from_xiulian()
        print("  ✓ 符号知识库初始化完成")
        
        models = await self.teacher_pool._get_session()
        available_models = await self.teacher_pool.query_teacher(
            "granite4:350m", "test", temperature=0.1, max_tokens=10
        )
        
        if "error" not in available_models:
            print("  ✓ 教师模型池连接成功")
            print(f"    可用教师: {list(self.teacher_pool.teachers.keys())}")
        else:
            print("  ⚠ 教师模型池连接异常，将使用离线模式")
        
        print("  ✓ 标准化测试基准加载完成")
        print(f"    测试用例: {len(self.benchmark.test_cases)}个")
        print(f"    测试类别: {list(self.benchmark.categories.keys())}")
        
        self._log_event("initialization", {"status": "completed"})
    
    async def run_baseline_evaluation(self) -> float:
        print("\n[阶段2] 基线评估...")
        
        results = self.benchmark.evaluate_student(self.student)
        baseline_score = self.benchmark.get_overall_score()
        baseline_pass_rate = self.benchmark.get_pass_rate()
        
        print(f"\n  基线得分: {baseline_score:.2%}")
        print(f"  基线通过率: {baseline_pass_rate:.2%}")
        
        cat_stats = self.benchmark.get_category_stats()
        print("\n  分类成绩:")
        for cat, stat in cat_stats.items():
            print(f"    {stat['description']}: {stat['score']:.2%}")
        
        self._log_event("baseline_evaluation", {
            "overall_score": baseline_score,
            "pass_rate": baseline_pass_rate,
            "category_stats": cat_stats
        })
        
        baseline_report = self.benchmark.generate_report()
        with open(self.output_dir / "baseline_report.md", 'w', encoding='utf-8') as f:
            f.write(baseline_report)
        
        self.benchmark.save_results(str(self.output_dir / "baseline_results.json"))
        
        return baseline_score
    
    async def training_iteration(self) -> TrainingProgress:
        self.iteration_count += 1
        
        print(f"\n[迭代 {self.iteration_count}]")
        
        progress = await self.training_loop.iteration_step()
        
        print(f"  教师: {progress.current_teacher}")
        print(f"  样本数: {progress.samples_generated}")
        print(f"  当前准确率: {progress.accuracy:.2%}")
        print(f"  当前损失: {progress.loss:.4f}")
        
        self._log_event("iteration", asdict(progress))
        
        if self.iteration_count % self.eval_interval == 0:
            await self._periodic_evaluation()
        
        if self.iteration_count % self.report_interval == 0:
            self._generate_progress_report()
        
        return progress
    
    async def _periodic_evaluation(self):
        print("\n  [定期评估]")
        
        overall, cat_stats = self.evaluator.evaluate_and_record()
        
        print(f"    评估得分: {overall:.2%}")
        print(f"    弱项领域: {', '.join(self.evaluator.get_weak_areas()) or '无'}")
        
        self._log_event("periodic_evaluation", {
            "overall_score": overall,
            "category_stats": cat_stats,
            "weak_areas": self.evaluator.get_weak_areas()
        })
        
        if overall > 0.85:
            print(f"\n  🎯 已达到优秀水平 ({overall:.2%})!")
    
    def _generate_progress_report(self):
        elapsed = time.time() - self.start_time
        iterations = self.iteration_count
        
        report_lines = [
            "# 融合模型训练进度报告",
            f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## 会话统计",
            f"- 运行时间: {elapsed/60:.1f}分钟",
            f"- 迭代次数: {iterations}",
            f"- 生成样本: {len(self.training_loop.samples)}",
            f"\n## 当前性能",
        ]
        
        if self.evaluator.evaluation_history:
            latest = self.evaluator.evaluation_history[-1]
            report_lines.extend([
                f"- 最新得分: {latest['overall_score']:.2%}",
                f"- 最佳得分: {latest['best_score']:.2%}",
                f"- 通过率: {latest['pass_rate']:.2%}",
            ])
        
        if self.training_loop.progress_history:
            latest_progress = self.training_loop.progress_history[-1]
            report_lines.extend([
                f"\n## 训练状态",
                f"- 当前阶段: {self.training_loop.current_phase.value}",
                f"- 当前损失: {latest_progress.loss:.4f}",
            ])
        
        report_lines.extend([
            f"\n## 教师贡献统计",
        ])
        for teacher, info in self.teacher_pool.teachers.items():
            report_lines.append(
                f"- {teacher}: {info['calls']}次调用 (权重={info['weight']})"
            )
        
        report_lines.extend([
            f"\n## 学生模型统计",
            f"- 符号推理调用: {self.student.stats['symbolic_calls']}",
            f"- 神经网络调用: {self.student.stats['neural_calls']}",
            f"- 混合推理调用: {self.student.stats['hybrid_calls']}",
            f"- 总调用次数: {self.student.stats['total_calls']}",
        ])
        
        report_content = '\n'.join(report_lines)
        
        with open(self.output_dir / "progress_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"    进度报告已更新")
    
    def _log_event(self, event_type: str, data: Dict):
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "iteration": self.iteration_count,
            "data": data
        }
        self.session_log.append(event)
    
    def _save_session_state(self):
        state = {
            "session_start": self.start_time,
            "session_end": time.time(),
            "iterations_completed": self.iteration_count,
            "running": self.running,
            "events": self.session_log[-100:],
            "evaluation_history": self.evaluator.evaluation_history,
            "best_score": self.evaluator.best_score,
            "student_stats": self.student.stats,
            "teacher_stats": self.teacher_pool.teachers,
        }
        
        with open(self.output_dir / "session_state.json", 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        self.student.save(str(self.output_dir))
        
        self.benchmark.save_results(str(self.output_dir / "final_results.json"))
        
        final_report = self.benchmark.generate_report()
        with open(self.output_dir / "final_report.md", 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"\n会话状态已保存到: {self.output_dir}")
    
    async def run(self):
        await self.initialize()
        
        baseline_score = await self.run_baseline_evaluation()
        
        print("\n[阶段3] 开始训练循环...")
        print(f"目标迭代次数: {self.max_iterations}")
        print(f"目标得分: 0.85 (优秀)")
        print(f"评估间隔: 每{self.eval_interval}次迭代")
        print(f"报告间隔: 每{self.report_interval}次迭代")
        print("\n按 Ctrl+C 可随时停止训练并保存进度")
        print("-"*70)
        
        while self.running and self.iteration_count < self.max_iterations:
            try:
                progress = await self.training_iteration()
                
                if progress.accuracy >= 0.85:
                    print(f"\n✅ 已达到目标准确率!")
                    break
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"\n⚠ 迭代错误: {e}")
                self._log_event("error", {"message": str(e)})
                await asyncio.sleep(5.0)
        
        print("\n[阶段4] 最终评估与保存...")
        
        final_score, _ = self.evaluator.evaluate_and_record()
        
        self._save_session_state()
        
        print("\n" + "="*70)
        print("训练会话总结")
        print("="*70)
        print(f"  运行时间: {(time.time() - self.start_time)/60:.1f}分钟")
        print(f"  完成迭代: {self.iteration_count}")
        print(f"  基线得分: {baseline_score:.2%}")
        print(f"  最终得分: {final_score:.2%}")
        print(f"  最佳得分: {self.evaluator.best_score:.2%}")
        print(f"  提升幅度: {(final_score - baseline_score):.2%}")
        print("="*70)
        
        await self.teacher_pool.close()
        
        return final_score


async def main():
    session = FusionTrainingSession(output_dir="fusion_paradigm/trained_model")
    
    try:
        final_score = await session.run()
        
        print(f"\n训练完成！最终得分: {final_score:.2%}")
        
        if final_score >= 0.85:
            print("🎉 达到优秀水平！")
        elif final_score >= 0.70:
            print("✓ 达到良好水平")
        else:
            print("⚠ 需要继续训练以达到目标水平")
            
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        session._save_session_state()
    except Exception as e:
        print(f"\n训练异常: {e}")
        session._save_session_state()


if __name__ == "__main__":
    asyncio.run(main())