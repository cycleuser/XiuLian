"""
快速训练演示脚本
简化版训练循环，快速迭代展示效果
"""

import asyncio
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_paradigm.fusion_training import (
    FusionStudentModel, OllamaTeacherPool
)


async def quick_train():
    print("="*60)
    print("融合范式模型 - 快速训练")
    print("="*60)
    
    output_dir = Path("fusion_paradigm/trained_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student = FusionStudentModel()
    student.bootstrap_from_xiulian()
    
    print("\n初始化完成")
    print("符号知识库规则:", len(student.symbolic_kb.rules))
    print("符号知识库模式:", len(student.symbolic_kb.patterns))
    print("符号知识库实体:", len(student.symbolic_kb.entities))
    
    teacher_pool = OllamaTeacherPool()
    
    test_questions = [
        ("什么是人工智能?", "qa"),
        ("调用echo msg=hello", "tool"),
        ("计算2+2", "math"),
        ("什么是机器学习?", "qa"),
        ("搜索深度学习", "search"),
    ]
    
    print("\n开始训练迭代...")
    print("-"*60)
    
    for iteration in range(1, 21):
        print(f"\n[迭代 {iteration}/20]")
        
        for question, category in test_questions:
            teacher = teacher_pool.get_next_teacher()
            
            print(f"  问题: {question[:20]}...")
            print(f"  教师: {teacher}")
            
            try:
                responses = await teacher_pool.query_all_teachers(question, temperature=0.7)
                best_response, confidence = teacher_pool.aggregate_responses(responses)
                
                if best_response:
                    from fusion_paradigm.fusion_training import TrainingSample
                    sample = TrainingSample(
                        input_text=question,
                        teacher_responses={m: r.get("response", "") 
                                          for m, r in responses.items() if "error" not in r},
                        best_response=best_response,
                        confidence=confidence,
                        category=category
                    )
                    student.learn_from_teacher(sample)
                    print(f"  学习完成，置信度: {confidence:.2f}")
                
            except Exception as e:
                print(f"  错误: {e}")
            
            await asyncio.sleep(0.1)
    
    print("\n" + "-"*60)
    print("保存模型...")
    student.save(str(output_dir))
    
    print("\n测试学生模型:")
    print("-"*60)
    
    for question, _ in test_questions:
        output = student.process(question)
        print(f"\nQ: {question}")
        print(f"A: {output.response}")
        print(f"方法: {output.method_used}, 置信度: {output.confidence:.2f}")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    await teacher_pool.close()


if __name__ == "__main__":
    asyncio.run(quick_train())