#!/usr/bin/env python3
from __future__ import annotations
"""
模糊逻辑实现

参考文献:
@article{zadeh1965,
  title={Fuzzy sets},
  author={Zadeh, Lotfi A.},
  journal={Information and Control},
  volume={8},
  number={3},
  pages={338--353},
  year={1965}
}

@article{zadeh1975,
  title={The concept of a linguistic variable and its application to approximate reasoning},
  author={Zadeh, Lotfi A.},
  journal={Information Sciences},
  volume={8},
  pages={199--249},
  year={1975}
}

@book{klir1995,
  title={Fuzzy Sets and Fuzzy Logic: Theory and Applications},
  author={Klir, George J. and Yuan, Bo},
  publisher={Prentice Hall},
  year={1995}
}

@article{mamdani1975,
  title={An experiment in linguistic synthesis with a fuzzy logic controller},
  author={Mamdani, Ebrahim H. and Assilian, Sedrak},
  journal={International Journal of Man-Machine Studies},
  volume={7},
  number={1},
  pages={1--13},
  year={1975}
}
"""

from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FuzzySet:
    name: str
    membership_func: Callable[[float], float]
    domain: Tuple[float, float]
    
    def membership(self, x: float) -> float:
        if self.domain[0] <= x <= self.domain[1]:
            return self.membership_func(x)
        return 0.0
    
    def __and__(self, other: FuzzySet) -> 'FuzzySet':
        return FuzzySet(
            f"{self.name} AND {other.name}",
            lambda x: min(self.membership(x), other.membership(x)),
            (max(self.domain[0], other.domain[0]), min(self.domain[1], other.domain[1]))
        )
    
    def __or__(self, other: FuzzySet) -> 'FuzzySet':
        return FuzzySet(
            f"{self.name} OR {other.name}",
            lambda x: max(self.membership(x), other.membership(x)),
            (min(self.domain[0], other.domain[0]), max(self.domain[1], other.domain[1]))
        )
    
    def __invert__(self) -> 'FuzzySet':
        return FuzzySet(
            f"NOT {self.name}",
            lambda x: 1.0 - self.membership(x),
            self.domain
        )


class FuzzyVariable:
    def __init__(self, name: str, domain: Tuple[float, float], sets: Dict[str, Callable[[float], float]]):
        self.name = name
        self.domain = domain
        self.fuzzy_sets: Dict[str, FuzzySet] = {}
        for set_name, func in sets.items():
            self.fuzzy_sets[set_name] = FuzzySet(set_name, func, domain)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        return {name: fs.membership(value) for name, fs in self.fuzzy_sets.items()}
    
    def get_set(self, name: str) -> FuzzySet:
        return self.fuzzy_sets[name]


class FuzzyRule:
    def __init__(self, antecedent: Dict[str, Tuple[str, str]], consequent: Tuple[str, str]):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        degrees = []
        for var_name, (set_name, op) in self.antecedent.items():
            if var_name in fuzzified_inputs:
                degree = fuzzified_inputs[var_name].get(set_name, 0.0)
                if op == 'NOT':
                    degree = 1.0 - degree
                degrees.append(degree)
        return min(degrees) if degrees else 0.0


class MamdaniFuzzySystem:
    """
    Mamdani模糊推理系统
    
    基于 Mamdani & Assilian (1975) 的模糊控制方法
    使用max-min合成进行推理
    """
    
    def __init__(self, input_vars: List[FuzzyVariable], output_vars: List[FuzzyVariable], rules: List[FuzzyRule]):
        self.input_vars = {v.name: v for v in input_vars}
        self.output_vars = {v.name: v for v in output_vars}
        self.rules = rules
    
    def infer(self, inputs: Dict[str, float]) -> Dict[str, float]:
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.input_vars:
                fuzzified[var_name] = self.input_vars[var_name].fuzzify(value)
        
        rule_outputs: Dict[str, Dict[str, float]] = {}
        for out_var in self.output_vars.values():
            rule_outputs[out_var.name] = {set_name: 0.0 for set_name in out_var.fuzzy_sets}
        
        for rule in self.rules:
            strength = rule.evaluate(fuzzified)
            out_var_name, out_set_name = rule.consequent
            if out_var_name in rule_outputs:
                rule_outputs[out_var_name][out_set_name] = max(
                    rule_outputs[out_var_name][out_set_name],
                    strength
                )
        
        results = {}
        for var_name, memberships in rule_outputs.items():
            results[var_name] = self._defuzzify_centroid(var_name, memberships)
        
        return results
    
    def _defuzzify_centroid(self, var_name: str, memberships: Dict[str, float]) -> float:
        out_var = self.output_vars[var_name]
        domain = out_var.domain
        
        total_area = 0.0
        weighted_sum = 0.0
        
        num_samples = 100
        for i in range(num_samples):
            x = domain[0] + (domain[1] - domain[0]) * i / (num_samples - 1)
            max_membership = 0.0
            for set_name, degree in memberships.items():
                fs = out_var.fuzzy_sets[set_name]
                clipped_membership = min(fs.membership(x), degree)
                max_membership = max(max_membership, clipped_membership)
            
            total_area += max_membership
            weighted_sum += x * max_membership
        
        return weighted_sum / total_area if total_area > 0 else 0.0


class FuzzyLogicController:
    """
    模糊逻辑控制器
    
    实现基于Zadeh (1965) 模糊集合理论的控制系统
    """
    
    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.system: MamdaniFuzzySystem = None
    
    def add_variable(self, name: str, domain: Tuple[float, float], 
                     triangular_sets: Dict[str, Tuple[float, float, float]]):
        """
        添加三角模糊集合变量
        
        Args:
            triangular_sets: {set_name: (a, b, c)} 三角函数参数
                             membership(x) = 0 if x <= a
                                           = (x-a)/(b-a) if a < x <= b
                                           = (c-x)/(c-b) if b < x < c
                                           = 0 if x >= c
        """
        def triangular(a, b, c):
            def func(x):
                if x <= a or x >= c:
                    return 0.0
                elif x <= b:
                    return (x - a) / (b - a) if b != a else 1.0
                else:
                    return (c - x) / (c - b) if c != b else 1.0
            return func
        
        sets = {name: triangular(a, b, c) for name, (a, b, c) in triangular_sets.items()}
        self.variables[name] = FuzzyVariable(name, domain, sets)
    
    def add_rule(self, antecedent: Dict[str, str], consequent: Tuple[str, str]):
        parsed_antecedent = {}
        for var, set_spec in antecedent.items():
            if set_spec.startswith('NOT '):
                parsed_antecedent[var] = (set_spec[4:], 'NOT')
            else:
                parsed_antecedent[var] = (set_spec, '')
        
        rule = FuzzyRule(parsed_antecedent, consequent)
        self.rules.append(rule)
    
    def build(self):
        input_vars = [v for v in self.variables.values()]
        output_vars = []
        
        output_var_names = set()
        for rule in self.rules:
            output_var_names.add(rule.consequent[0])
        
        for var_name in output_var_names:
            if var_name in self.variables:
                output_vars.append(self.variables[var_name])
        
        self.system = MamdaniFuzzySystem(input_vars, output_vars, self.rules)
    
    def control(self, inputs: Dict[str, float]) -> Dict[str, float]:
        if self.system is None:
            self.build()
        return self.system.infer(inputs)


def create_temperature_controller() -> FuzzyLogicController:
    """
    创建温度模糊控制器示例
    
    典型应用: 空调温度控制
    """
    controller = FuzzyLogicController()
    
    controller.add_variable('temperature', (0, 40), {
        'cold': (0, 0, 15),
        'cool': (5, 15, 25),
        'warm': (15, 25, 35),
        'hot': (25, 40, 40)
    })
    
    controller.add_variable('humidity', (0, 100), {
        'low': (0, 0, 40),
        'medium': (20, 50, 80),
        'high': (60, 100, 100)
    })
    
    controller.add_variable('fan_speed', (0, 100), {
        'slow': (0, 0, 30),
        'medium': (20, 50, 80),
        'fast': (70, 100, 100)
    })
    
    controller.add_rule({'temperature': 'cold'}, ('fan_speed', 'slow'))
    controller.add_rule({'temperature': 'cool'}, ('fan_speed', 'medium'))
    controller.add_rule({'temperature': 'warm'}, ('fan_speed', 'medium'))
    controller.add_rule({'temperature': 'hot'}, ('fan_speed', 'fast'))
    controller.add_rule({'temperature': 'hot', 'humidity': 'high'}, ('fan_speed', 'fast'))
    
    controller.build()
    
    return controller


class FuzzySimilarity:
    """
    模糊相似度计算
    
    基于 Zadeh (1975) 的模糊集合运算
    """
    
    @staticmethod
    def jaccard(set_a: Dict[str, float], set_b: Dict[str, float]) -> float:
        intersection = sum(min(set_a.get(k, 0), set_b.get(k, 0)) for k in set_a.keys() | set_b.keys())
        union = sum(max(set_a.get(k, 0), set_b.get(k, 0)) for k in set_a.keys() | set_b.keys())
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def cosine(set_a: Dict[str, float], set_b: Dict[str, float]) -> float:
        dot = sum(set_a.get(k, 0) * set_b.get(k, 0) for k in set_a.keys() | set_b.keys())
        norm_a = sum(v**2 for v in set_a.values()) ** 0.5
        norm_b = sum(v**2 for v in set_b.values()) ** 0.5
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


if __name__ == "__main__":
    controller = create_temperature_controller()
    
    print("="*60)
    print("模糊逻辑控制器测试")
    print("="*60)
    
    test_cases = [
        {'temperature': 10, 'humidity': 30},
        {'temperature': 20, 'humidity': 50},
        {'temperature': 35, 'humidity': 80},
    ]
    
    for case in test_cases:
        result = controller.control(case)
        print(f"\n输入: 温度={case['temperature']}°C, 湿度={case['humidity']}%")
        print(f"输出: 风扇速度={result.get('fan_speed', 0):.1f}%")
    
    print("\n" + "="*60)
    print("模糊相似度测试")
    print("="*60)
    
    set_a = {'hot': 0.8, 'warm': 0.2, 'cool': 0.0}
    set_b = {'hot': 0.6, 'warm': 0.4, 'cool': 0.0}
    
    print(f"\n集合A: {set_a}")
    print(f"集合B: {set_b}")
    print(f"Jaccard相似度: {FuzzySimilarity.jaccard(set_a, set_b):.3f}")
    print(f"Cosine相似度: {FuzzySimilarity.cosine(set_a, set_b):.3f}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Zadeh (1965) - 模糊集合理论奠基论文")
    print("Mamdani & Assilian (1975) - 首个模糊控制系统")
    print("Zadeh (1975) - 语言变量与近似推理")