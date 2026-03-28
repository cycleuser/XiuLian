"""
符号主义方法实现

参考文献:
- Feigenbaum (1970): 专家系统
- Newell & Simon (1972): 产生式系统  
- Kolodner (1992): 案例推理
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import re


@dataclass
class Rule:
    """产生式规则
    
    基于Newell & Simon (1972)的产生式系统理论
    """
    condition: Dict[str, Any]  # IF部分
    action: Dict[str, Any]      # THEN部分
    priority: int = 0           # 冲突解决优先级
    confidence: float = 1.0     # 置信度(用于不确定性推理)


class ExpertSystem:
    """专家系统实现
    
    基于Feigenbaum et al. (1970)的经典专家系统架构
    
    参考:
    - Feigenbaum, E.A. & Buchanan, B.G. (1970). "A comparative study of interpretation systems"
    - Shortliffe, E.H. (1975). "A model of inexact reasoning in medicine" (MYCIN系统)
    
    时间复杂度:
    - 规则匹配: O(n*m) 其中n为规则数，m为事实数
    - 推理: O(k) 其中k为推理链长度
    """
    
    def __init__(self):
        self.knowledge_base: List[Rule] = []
        self.working_memory: Dict[str, Any] = {}
        self.inference_log: List[str] = []
    
    def add_rule(self, condition: Dict, action: Dict, priority: int = 0, confidence: float = 1.0):
        """添加规则到知识库"""
        rule = Rule(condition, action, priority, confidence)
        self.knowledge_base.append(rule)
        # 按优先级排序
        self.knowledge_base.sort(key=lambda r: -r.priority)
    
    def add_fact(self, key: str, value: Any):
        """添加事实到工作记忆"""
        self.working_memory[key] = value
    
    def match_rule(self, rule: Rule) -> bool:
        """规则匹配 - O(m)
        
        检查规则条件是否满足
        """
        for key, expected_value in rule.condition.items():
            if key not in self.working_memory:
                return False
            actual_value = self.working_memory[key]
            
            # 支持多种匹配模式
            if isinstance(expected_value, str) and expected_value.startswith('~'):
                # 非匹配
                if actual_value == expected_value[1:]:
                    return False
            elif isinstance(expected_value, (list, set)):
                # 集合匹配
                if actual_value not in expected_value:
                    return False
            elif callable(expected_value):
                # 函数匹配
                if not expected_value(actual_value):
                    return False
            else:
                # 精确匹配
                if actual_value != expected_value:
                    return False
        
        return True
    
    def execute_rule(self, rule: Rule):
        """执行规则动作"""
        for key, value in rule.action.items():
            if callable(value):
                # 如果是函数，传入工作记忆
                self.working_memory[key] = value(self.working_memory)
            else:
                self.working_memory[key] = value
        
        self.inference_log.append(f"Applied rule: {rule.condition} -> {rule.action}")
    
    def forward_chaining(self, max_iterations: int = 100) -> bool:
        """前向链推理
        
        基于数据驱动的推理方式
        从已知事实出发，应用规则推导新事实
        
        时间复杂度: O(n*k) 其中n为规则数，k为迭代次数
        """
        changed = True
        iterations = 0
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            # 冲突解决：选择最高优先级的匹配规则
            for rule in self.knowledge_base:
                if self.match_rule(rule):
                    # 执行规则前检查是否会产生新事实
                    produces_new = False
                    for key in rule.action.keys():
                        if key not in self.working_memory or self.working_memory[key] != rule.action[key]:
                            produces_new = True
                            break
                    
                    if produces_new:
                        self.execute_rule(rule)
                        changed = True
                        break  # 一次只执行一条规则
        
        return iterations < max_iterations
    
    def backward_chaining(self, goal: str, visited: set = None) -> Tuple[bool, float]:
        """后向链推理
        
        基于目标驱动的推理方式
        从目标出发，反向寻找支持证据
        
        参考: PROLOG的推理机制
        
        时间复杂度: O(2^n) 最坏情况，实际取决于规则结构
        """
        if visited is None:
            visited = set()
        
        # 检查目标是否已知
        if goal in self.working_memory:
            return True, 1.0
        
        # 防止循环
        if goal in visited:
            return False, 0.0
        
        visited.add(goal)
        
        # 寻找能推导目标的规则
        for rule in self.knowledge_base:
            if goal in rule.action:
                # 递归检查条件
                all_conditions_met = True
                total_confidence = rule.confidence
                
                for key, value in rule.condition.items():
                    sub_goal = f"{key}={value}"
                    met, confidence = self.backward_chaining(sub_goal, visited.copy())
                    
                    if not met:
                        all_conditions_met = False
                        break
                    
                    total_confidence *= confidence
                
                if all_conditions_met:
                    return True, total_confidence
        
        return False, 0.0
    
    def explain(self) -> List[str]:
        """解释推理过程
        
        专家系统的核心优势：可解释性
        """
        return self.inference_log


class ProductionSystem:
    """产生式系统实现
    
    基于Newell & Simon (1972)的人类问题解决理论
    
    参考:
    - Newell, A. & Simon, H.A. (1972). "Human problem solving"
    - Forgy, C.L. (1979). "On the efficient implementation of production systems"
    - OPS5: Brownston et al. (1985). "Programming expert systems in OPS5"
    
    架构:
    1. 产生式记忆 - 规则库
    2. 工作记忆 - 事实库
    3. 识别-动作循环
    
    时间复杂度:
    - Rete算法: O(n) 匹配，其中n为事实数
    - 简单实现: O(n*m) 其中n为规则数，m为事实数
    """
    
    def __init__(self):
        self.productions: List[Tuple[callable, callable]] = []  # (条件, 动作)
        self.wm: List[Dict] = []  # 工作记忆元素
        self.conflict_set: List[int] = []  # 冲突集
    
    def add_production(self, condition: callable, action: callable):
        """添加产生式规则"""
        self.productions.append((condition, action))
    
    def add_wme(self, wme: Dict):
        """添加工作记忆元素"""
        self.wm.append(wme)
    
    def match(self) -> List[int]:
        """模式匹配阶段
        
        找出所有条件满足的产生式
        """
        self.conflict_set = []
        
        for i, (condition, _) in enumerate(self.productions):
            if condition(self.wm):
                self.conflict_set.append(i)
        
        return self.conflict_set
    
    def resolve(self) -> int:
        """冲突解决阶段
        
        选择一个产生式执行
        
        策略:
        1. 折射性 - 新近激活的规则优先
        2. 特殊性 - 更具体的规则优先  
        3. 任意性 - 随机选择
        """
        if not self.conflict_set:
            return -1
        
        # 简单实现：选择第一个
        return self.conflict_set[0]
    
    def act(self, production_index: int):
        """执行阶段
        
        执行选定产生式的动作
        """
        if 0 <= production_index < len(self.productions):
            _, action = self.productions[production_index]
            action(self.wm)
    
    def recognize_act_cycle(self, max_cycles: int = 100) -> int:
        """识别-动作循环
        
        产生式系统的核心执行循环
        
        返回执行的循环次数
        """
        for cycle in range(max_cycles):
            # 识别阶段
            self.match()
            
            if not self.conflict_set:
                return cycle  # 没有可执行的规则
            
            # 冲突解决
            selected = self.resolve()
            
            # 动作阶段
            self.act(selected)
        
        return max_cycles


class CaseBasedReasoning:
    """案例推理实现
    
    基于Schank (1982)和Kolodner (1992)的CBR理论
    
    参考:
    - Schank, R.C. (1982). "Dynamic Memory: A Theory of Reminding and Learning"
    - Kolodner, J.L. (1992). "An introduction to case-based reasoning"
    - Aamodt, A. & Plaza, E. (1994). "Case-Based Reasoning: Foundational Issues"
    
    CBR循环:
    1. RETRIEVE - 检索相似案例
    2. REUSE - 重用案例解决方案
    3. REVISE - 修订解决方案
    4. RETAIN - 保存新案例
    
    时间复杂度:
    - 线性检索: O(n)
    - 索引检索: O(log n)
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.case_base: List[Dict] = []
        self.similarity_threshold = similarity_threshold
        self.index: Dict[str, List[int]] = {}  # 索引加速检索
    
    def add_case(self, problem: Dict, solution: Dict, outcome: Dict = None):
        """添加案例到案例库"""
        case = {
            'problem': problem,
            'solution': solution,
            'outcome': outcome,
            'id': len(self.case_base)
        }
        self.case_base.append(case)
        
        # 建立索引
        for key, value in problem.items():
            index_key = f"{key}:{value}"
            if index_key not in self.index:
                self.index[index_key] = []
            self.index[index_key].append(case['id'])
    
    def similarity(self, problem1: Dict, problem2: Dict) -> float:
        """计算问题相似度
        
        使用特征重叠度计算
        
        时间复杂度: O(k) 其中k为特征数
        """
        if not problem1 or not problem2:
            return 0.0
        
        common_keys = set(problem1.keys()) & set(problem2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if problem1[k] == problem2[k])
        
        # Jaccard相似度
        total_keys = set(problem1.keys()) | set(problem2.keys())
        
        return matches / len(total_keys)
    
    def retrieve(self, new_problem: Dict, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索阶段 - 找出最相似案例
        
        时间复杂度: 
        - 线性扫描: O(n*k)
        - 索引加速: O(m*k) 其中m为候选案例数
        """
        candidates = []
        
        # 使用索引加速
        candidate_ids = set()
        for key, value in new_problem.items():
            index_key = f"{key}:{value}"
            if index_key in self.index:
                candidate_ids.update(self.index[index_key])
        
        # 如果索引无结果，全量扫描
        if not candidate_ids:
            candidate_ids = range(len(self.case_base))
        
        # 计算相似度
        for case_id in candidate_ids:
            case = self.case_base[case_id]
            sim = self.similarity(new_problem, case['problem'])
            if sim >= self.similarity_threshold:
                candidates.append((case, sim))
        
        # 排序返回top-k
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]
    
    def reuse(self, retrieved_case: Dict, new_problem: Dict) -> Dict:
        """重用阶段 - 适配解决方案
        
        简单实现：直接重用或参数替换
        """
        solution = retrieved_case['solution'].copy()
        
        # 参数替换
        if 'params' in solution:
            for key, value in new_problem.items():
                if key in solution['params']:
                    solution['params'][key] = value
        
        return solution
    
    def revise(self, solution: Dict, new_problem: Dict, feedback: Dict = None) -> Dict:
        """修订阶段 - 根据反馈调整方案
        
        实际应用中可能需要领域知识
        """
        if feedback and 'adjustments' in feedback:
            for key, value in feedback['adjustments'].items():
                solution[key] = value
        
        return solution
    
    def retain(self, problem: Dict, solution: Dict, outcome: Dict = None):
        """保存阶段 - 学习新案例
        
        将成功案例加入案例库
        """
        if outcome is None or outcome.get('success', True):
            self.add_case(problem, solution, outcome)
    
    def solve(self, problem: Dict, feedback: Dict = None) -> Tuple[Dict, float]:
        """完整CBR循环"""
        # 1. Retrieve
        similar_cases = self.retrieve(problem, top_k=1)
        
        if not similar_cases:
            # 没有相似案例，返回默认方案
            return {'solution': 'unknown', 'confidence': 0.0}, 0.0
        
        best_case, similarity = similar_cases[0]
        
        # 2. Reuse
        solution = self.reuse(best_case, problem)
        
        # 3. Revise
        if feedback:
            solution = self.revise(solution, problem, feedback)
        
        # 4. Retain
        self.retain(problem, solution, feedback)
        
        return solution, similarity


# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("符号主义方法测试")
    print("="*60)
    
    # 1. 专家系统测试
    print("\n1. 专家系统 (Expert System)")
    print("-"*40)
    
    es = ExpertSystem()
    
    # 添加规则
    es.add_rule(
        condition={'temperature': lambda x: x > 38},
        action={'diagnosis': 'fever', 'treatment': 'rest'}
    )
    es.add_rule(
        condition={'diagnosis': 'fever', 'cough': True},
        action={'diagnosis': 'flu', 'treatment': 'antiviral'}
    )
    
    # 添加事实
    es.add_fact('temperature', 39)
    es.add_fact('cough', True)
    
    # 推理
    es.forward_chaining()
    
    print(f"诊断结果: {es.working_memory.get('diagnosis')}")
    print(f"治疗方案: {es.working_memory.get('treatment')}")
    print(f"推理过程: {es.explain()}")
    
    # 2. 产生式系统测试
    print("\n2. 产生式系统 (Production System)")
    print("-"*40)
    
    ps = ProductionSystem()
    
    # 添加产生式
    ps.add_production(
        condition=lambda wm: any(w.get('type') == 'bird' for w in wm),
        action=lambda wm: wm.append({'type': 'can-fly', 'source': 'bird-rule'})
    )
    
    # 添加工作记忆
    ps.add_wme({'type': 'bird', 'name': 'sparrow'})
    
    # 运行
    cycles = ps.recognize_act_cycle(max_cycles=10)
    print(f"执行循环数: {cycles}")
    print(f"工作记忆: {ps.wm}")
    
    # 3. 案例推理测试
    print("\n3. 案例推理 (Case-Based Reasoning)")
    print("-"*40)
    
    cbr = CaseBasedReasoning(similarity_threshold=0.5)
    
    # 添加案例
    cbr.add_case(
        problem={'color': 'red', 'shape': 'round', 'size': 'medium'},
        solution={'fruit': 'apple'}
    )
    cbr.add_case(
        problem={'color': 'yellow', 'shape': 'long', 'size': 'medium'},
        solution={'fruit': 'banana'}
    )
    
    # 新问题
    new_problem = {'color': 'red', 'shape': 'round', 'size': 'large'}
    solution, similarity = cbr.solve(new_problem)
    
    print(f"新问题: {new_problem}")
    print(f"最相似案例相似度: {similarity:.2f}")
    print(f"解决方案: {solution}")