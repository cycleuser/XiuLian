"""
标准化测试评估框架
评估融合模型在标准测试上的表现
"""

import asyncio
import json
import time
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_paradigm.fusion_training import (
    FusionStudentModel, OllamaTeacherPool, TrainingSample
)


@dataclass
class TestCase:
    id: str
    category: str
    input_text: str
    expected_output: str
    expected_keywords: List[str]
    difficulty: float = 1.0
    reference_answer: str = ""
    rubric: Dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    test_id: str
    category: str
    student_output: str
    expected_output: str
    accuracy_score: float
    keyword_coverage: float
    semantic_similarity: float
    overall_score: float
    passed: bool
    feedback: str
    latency_ms: float


class StandardBenchmark:
    """标准化基准测试
    
    测试类别：
    1. 工具调用准确性
    2. 知识问答准确性  
    3. 搜索推理准确性
    4. 数学计算准确性
    5. 语义理解准确性
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[EvaluationResult] = []
        self.categories = {
            "tool_call": {"weight": 0.25, "description": "工具调用准确性"},
            "qa": {"weight": 0.30, "description": "知识问答准确性"},
            "search": {"weight": 0.15, "description": "搜索推理准确性"},
            "math": {"weight": 0.15, "description": "数学计算准确性"},
            "semantic": {"weight": 0.15, "description": "语义理解准确性"},
        }
        
        self._load_standard_tests()
    
    def _load_standard_tests(self):
        self.test_cases = [
            TestCase(
                id="tool_001",
                category="tool_call",
                input_text="调用echo msg=hello",
                expected_output="{'message': 'hello'}",
                expected_keywords=["hello", "message"],
                difficulty=0.3,
                rubric={"exact_match": 1.0, "keyword_match": 0.8}
            ),
            TestCase(
                id="tool_002",
                category="tool_call",
                input_text="计算2+2",
                expected_output="4",
                expected_keywords=["4"],
                difficulty=0.2,
                rubric={"exact_match": 1.0, "keyword_match": 0.9}
            ),
            TestCase(
                id="tool_003",
                category="tool_call",
                input_text="调用calc expr=10*5",
                expected_output="{'result': 50}",
                expected_keywords=["50", "result"],
                difficulty=0.3,
                rubric={"exact_match": 1.0, "keyword_match": 0.8}
            ),
            TestCase(
                id="qa_001",
                category="qa",
                input_text="什么是人工智能?",
                expected_output="人工智能是模拟人类智能的技术领域",
                expected_keywords=["智能", "技术", "模拟", "AI"],
                difficulty=0.5,
                reference_answer="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行需要人类智能的任务的系统。",
                rubric={"keyword_coverage": 0.5, "semantic_similarity": 0.5}
            ),
            TestCase(
                id="qa_002",
                category="qa",
                input_text="什么是机器学习?",
                expected_output="机器学习是让计算机从数据中学习的方法",
                expected_keywords=["机器学习", "数据", "学习", "算法"],
                difficulty=0.5,
                reference_answer="机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式并做出预测。",
                rubric={"keyword_coverage": 0.5, "semantic_similarity": 0.5}
            ),
            TestCase(
                id="qa_003",
                category="qa",
                input_text="深度学习和机器学习的区别?",
                expected_output="深度学习是机器学习的子集，使用神经网络",
                expected_keywords=["深度学习", "机器学习", "神经网络", "子集", "区别"],
                difficulty=0.7,
                reference_answer="深度学习是机器学习的一个专门分支，使用多层神经网络进行特征学习和模式识别。",
                rubric={"keyword_coverage": 0.4, "semantic_similarity": 0.6}
            ),
            TestCase(
                id="qa_004",
                category="qa",
                input_text="NLP是什么?",
                expected_output="自然语言处理是让计算机理解人类语言",
                expected_keywords=["自然", "语言", "处理", "NLP", "理解"],
                difficulty=0.6,
                reference_answer="自然语言处理（NLP）是人工智能的一个领域，专注于让计算机理解和生成人类语言。",
                rubric={"keyword_coverage": 0.5, "semantic_similarity": 0.5}
            ),
            TestCase(
                id="search_001",
                category="search",
                input_text="搜索人工智能",
                expected_output="关于人工智能的信息",
                expected_keywords=["人工智能", "信息", "AI"],
                difficulty=0.4,
                rubric={"keyword_coverage": 0.7, "relevance": 0.3}
            ),
            TestCase(
                id="search_002",
                category="search",
                input_text="查找机器学习的应用",
                expected_output="机器学习应用于图像识别、推荐系统等",
                expected_keywords=["应用", "机器学习", "图像", "推荐"],
                difficulty=0.6,
                rubric={"keyword_coverage": 0.6, "relevance": 0.4}
            ),
            TestCase(
                id="math_001",
                category="math",
                input_text="计算100+50",
                expected_output="150",
                expected_keywords=["150"],
                difficulty=0.1,
                rubric={"exact_match": 1.0}
            ),
            TestCase(
                id="math_002",
                category="math",
                input_text="计算25*4",
                expected_output="100",
                expected_keywords=["100"],
                difficulty=0.2,
                rubric={"exact_match": 1.0}
            ),
            TestCase(
                id="math_003",
                category="math",
                input_text="计算100/5",
                expected_output="20",
                expected_keywords=["20"],
                difficulty=0.2,
                rubric={"exact_match": 1.0}
            ),
            TestCase(
                id="semantic_001",
                category="semantic",
                input_text="苹果公司发明了iPhone",
                expected_output="是的，苹果公司是iPhone的制造商",
                expected_keywords=["苹果", "iPhone", "公司", "制造商"],
                difficulty=0.6,
                rubric={"semantic_similarity": 0.6, "keyword_coverage": 0.4}
            ),
            TestCase(
                id="semantic_002",
                category="semantic",
                input_text="Python是一种编程语言",
                expected_output="是的，Python是流行的编程语言",
                expected_keywords=["Python", "编程", "语言"],
                difficulty=0.5,
                rubric={"semantic_similarity": 0.5, "keyword_coverage": 0.5}
            ),
            TestCase(
                id="semantic_003",
                category="semantic",
                input_text="机器学习需要大量数据训练",
                expected_output="数据是机器学习训练的关键",
                expected_keywords=["数据", "训练", "机器学习"],
                difficulty=0.7,
                rubric={"semantic_similarity": 0.6, "keyword_coverage": 0.4}
            ),
        ]
    
    def evaluate_student(self, student: FusionStudentModel) -> List[EvaluationResult]:
        self.results = []
        
        for test in self.test_cases:
            start_time = time.time()
            
            output = student.process(test.input_text)
            
            student_output = output.response
            
            keyword_coverage = self._calc_keyword_coverage(
                student_output, test.expected_keywords
            )
            
            semantic_similarity = self._calc_semantic_similarity(
                student_output, test.reference_answer or test.expected_output
            )
            
            accuracy_score = self._calc_accuracy(
                student_output, test.expected_output, test.rubric
            )
            
            overall_score = self._calc_overall(
                keyword_coverage, semantic_similarity, accuracy_score, test.rubric
            )
            
            passed = overall_score >= 0.7
            
            feedback = self._generate_feedback(
                test, student_output, keyword_coverage, semantic_similarity
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = EvaluationResult(
                test_id=test.id,
                category=test.category,
                student_output=student_output,
                expected_output=test.expected_output,
                accuracy_score=accuracy_score,
                keyword_coverage=keyword_coverage,
                semantic_similarity=semantic_similarity,
                overall_score=overall_score,
                passed=passed,
                feedback=feedback,
                latency_ms=latency_ms
            )
            
            self.results.append(result)
        
        return self.results
    
    def _calc_keyword_coverage(self, output: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        
        output_lower = output.lower()
        found = sum(1 for kw in keywords if kw.lower() in output_lower)
        
        return found / len(keywords)
    
    def _calc_semantic_similarity(self, output: str, reference: str) -> float:
        output_words = set(output.lower().split())
        ref_words = set(reference.lower().split())
        
        if not output_words or not ref_words:
            return 0.0
        
        overlap = len(output_words & ref_words)
        union = len(output_words | ref_words)
        
        jaccard = overlap / union if union > 0 else 0
        
        output_len = len(output)
        ref_len = len(reference)
        len_ratio = min(output_len, ref_len) / max(output_len, ref_len)
        
        return 0.7 * jaccard + 0.3 * len_ratio
    
    def _calc_accuracy(self, output: str, expected: str, rubric: Dict) -> float:
        output_lower = output.lower().strip()
        expected_lower = expected.lower().strip()
        
        if output_lower == expected_lower:
            return rubric.get("exact_match", 1.0)
        
        if expected_lower in output_lower or output_lower in expected_lower:
            return rubric.get("keyword_match", 0.8)
        
        return 0.0
    
    def _calc_overall(self, keyword_cov: float, semantic_sim: float, 
                       accuracy: float, rubric: Dict) -> float:
        if "exact_match" in rubric:
            return accuracy
        
        kw_weight = rubric.get("keyword_coverage", 0.3)
        sem_weight = rubric.get("semantic_similarity", 0.3)
        acc_weight = rubric.get("accuracy", 0.4) if accuracy > 0 else 0
        
        total_weight = kw_weight + sem_weight + acc_weight
        if total_weight == 0:
            total_weight = 1
        
        return (keyword_cov * kw_weight + semantic_sim * sem_weight + accuracy * acc_weight) / total_weight
    
    def _generate_feedback(self, test: TestCase, output: str,
                           keyword_cov: float, semantic_sim: float) -> str:
        if keyword_cov >= 0.8 and semantic_sim >= 0.6:
            return "优秀回答"
        
        if keyword_cov >= 0.5:
            missing = [kw for kw in test.expected_keywords 
                      if kw.lower() not in output.lower()]
            return f"部分正确，缺少关键词: {', '.join(missing[:3])}"
        
        return f"需要改进，参考答案: {test.reference_answer[:50]}..."
    
    def get_category_stats(self) -> Dict[str, Dict]:
        stats = {}
        
        for cat, info in self.categories.items():
            cat_results = [r for r in self.results if r.category == cat]
            
            if not cat_results:
                stats[cat] = {"score": 0, "pass_rate": 0, "count": 0}
                continue
            
            avg_score = sum(r.overall_score for r in cat_results) / len(cat_results)
            pass_rate = sum(1 for r in cat_results if r.passed) / len(cat_results)
            avg_latency = sum(r.latency_ms for r in cat_results) / len(cat_results)
            
            stats[cat] = {
                "score": avg_score,
                "pass_rate": pass_rate,
                "count": len(cat_results),
                "avg_latency_ms": avg_latency,
                "description": info["description"]
            }
        
        return stats
    
    def get_overall_score(self) -> float:
        if not self.results:
            return 0.0
        
        weighted_score = 0
        total_weight = 0
        
        for cat, info in self.categories.items():
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                cat_score = sum(r.overall_score for r in cat_results) / len(cat_results)
                weighted_score += cat_score * info["weight"]
                total_weight += info["weight"]
        
        return weighted_score / total_weight if total_weight > 0 else 0
    
    def get_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        
        passed = sum(1 for r in self.results if r.passed)
        return passed / len(self.results)
    
    def generate_report(self) -> str:
        lines = [
            "# 标准化测试评估报告",
            f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## 总体成绩",
            f"- 总体得分: **{self.get_overall_score():.2%}**",
            f"- 通过率: **{self.get_pass_rate():.2%}**",
            f"- 测试总数: {len(self.results)}",
            f"\n## 分类成绩",
        ]
        
        stats = self.get_category_stats()
        for cat, stat in stats.items():
            lines.extend([
                f"\n### {stat['description']}",
                f"- 得分: {stat['score']:.2%}",
                f"- 通过率: {stat['pass_rate']:.2%}",
                f"- 测试数: {stat['count']}",
                f"- 平均延迟: {stat['avg_latency_ms']:.2f}ms",
            ])
        
        lines.append("\n## 详细结果")
        for r in self.results:
            status = "✅ 通过" if r.passed else "❌ 未通过"
            lines.extend([
                f"\n### {r.test_id} - {r.category} {status}",
                f"- 输入: `{r.student_output}`",
                f"- 期望: `{r.expected_output}`",
                f"- 关词覆盖: {r.keyword_coverage:.2%}",
                f"- 语义相似: {r.semantic_similarity:.2%}",
                f"- 总分: {r.overall_score:.2%}",
                f"- 反馈: {r.feedback}",
            ])
        
        lines.append("\n## 改进建议")
        
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            lines.append(f"\n需要改进的测试 ({len(failed_tests)}个):")
            for r in failed_tests[:5]:
                lines.append(f"- {r.test_id}: {r.feedback}")
        
        low_categories = [cat for cat, stat in stats.items() if stat["score"] < 0.7]
        if low_categories:
            lines.append(f"\n需要加强的类别:")
            for cat in low_categories:
                lines.append(f"- {stats[cat]['description']}: 当前得分 {stats[cat]['score']:.2%}")
        
        return '\n'.join(lines)
    
    def save_results(self, path: str):
        data = {
            "timestamp": time.time(),
            "overall_score": self.get_overall_score(),
            "pass_rate": self.get_pass_rate(),
            "category_stats": self.get_category_stats(),
            "results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "student_output": r.student_output,
                    "expected_output": r.expected_output,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "feedback": r.feedback,
                    "latency_ms": r.latency_ms
                }
                for r in self.results
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class ContinuousEvaluation:
    """持续评估与反馈
    
    在训练过程中持续评估，提供反馈用于改进
    """
    
    def __init__(self, student: FusionStudentModel, benchmark: StandardBenchmark):
        self.student = student
        self.benchmark = benchmark
        self.evaluation_history: List[Dict] = []
        self.best_score = 0.0
    
    def evaluate_and_record(self) -> Tuple[float, Dict]:
        results = self.benchmark.evaluate_student(self.student)
        
        overall = self.benchmark.get_overall_score()
        pass_rate = self.benchmark.get_pass_rate()
        cat_stats = self.benchmark.get_category_stats()
        
        if overall > self.best_score:
            self.best_score = overall
        
        record = {
            "timestamp": time.time(),
            "overall_score": overall,
            "pass_rate": pass_rate,
            "best_score": self.best_score,
            "improvement": overall - (self.evaluation_history[-1]["overall_score"] 
                                      if self.evaluation_history else 0),
            "category_stats": cat_stats
        }
        
        self.evaluation_history.append(record)
        
        return overall, cat_stats
    
    def get_weak_areas(self) -> List[str]:
        cat_stats = self.benchmark.get_category_stats()
        weak = [cat for cat, stat in cat_stats.items() if stat["score"] < 0.7]
        return weak
    
    def generate_training_samples_for_weak_areas(
        self, teacher_pool: OllamaTeacherPool, num_samples: int = 5
    ) -> List[TrainingSample]:
        weak_cats = self.get_weak_areas()
        samples = []
        
        weak_tests = [t for t in self.benchmark.test_cases 
                      if t.category in weak_cats]
        
        for test in weak_tests[:num_samples]:
            prompt = f"请回答以下问题并给出准确答案：\n{test.input_text}"
            
            async def get_sample():
                responses = await teacher_pool.query_all_teachers(prompt)
                best, confidence = teacher_pool.aggregate_responses(responses)
                return TrainingSample(
                    input_text=test.input_text,
                    teacher_responses={m: r.get("response", "") 
                                       for m, r in responses.items() if "error" not in r},
                    best_response=best,
                    confidence=confidence,
                    category=test.category,
                    metadata={"purpose": "weak_area_improvement"}
                )
            
            sample = asyncio.get_event_loop().run_until_complete(get_sample())
            samples.append(sample)
        
        return samples
    
    def get_progress_summary(self) -> str:
        if not self.evaluation_history:
            return "尚未进行评估"
        
        latest = self.evaluation_history[-1]
        first = self.evaluation_history[0]
        
        improvement = latest["overall_score"] - first["overall_score"]
        
        lines = [
            "## 评估进度摘要",
            f"- 当前得分: {latest['overall_score']:.2%}",
            f"- 最佳得分: {self.best_score:.2%}",
            f"- 累计提升: {improvement:.2%}",
            f"- 评估次数: {len(self.evaluation_history)}",
            f"- 弱项领域: {', '.join(self.get_weak_areas()) or '无'}",
        ]
        
        return '\n'.join(lines)


async def run_evaluation():
    student = FusionStudentModel()
    student.bootstrap_from_xiulian()
    
    benchmark = StandardBenchmark()
    
    results = benchmark.evaluate_student(student)
    
    print(benchmark.generate_report())
    
    benchmark.save_results("fusion_paradigm/evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(run_evaluation())