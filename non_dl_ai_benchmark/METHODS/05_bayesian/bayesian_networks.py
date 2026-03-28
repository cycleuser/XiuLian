#!/usr/bin/env python3
"""
贝叶斯方法实现

参考文献:
@article{maron1961,
  title={Automatic indexing: An experimental inquiry},
  author={Maron, M. E.},
  journal={Journal of the ACM},
  volume={8},
  number={3},
  pages={404--417},
  year={1961}
}

@book{pearl1988,
  title={Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference},
  author={Pearl, Judea},
  publisher={Morgan Kaufmann},
  year={1988}
}

@article{heckerman1995,
  title={A tutorial on learning with Bayesian networks},
  author={Heckerman, David},
  journal={Microsoft Research},
  year={1995}
}

@article{lewis1998,
  title={Naive (Bayes) at forty: The independence assumption in information retrieval},
  author={Lewis, David D.},
  booktitle={Machine Learning: ECML-98},
  pages={4--15},
  year={1998}
}
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class BayesianNode:
    name: str
    parents: List[str] = field(default_factory=list)
    cpt: Dict[Tuple[str, ...], float] = field(default_factory=dict)


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器
    
    基于 Maron (1961) 和 Lewis (1998) 的方法
    假设特征独立: P(x1,...,xn|C) = ∏ P(xi|C)
    """
    
    def __init__(self):
        self.class_prior: Dict[str, float] = {}
        self.feature_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.classes: Set[str] = set()
        self.features: Set[str] = set()
        self.smoothing: float = 1.0
    
    def fit(self, data: List[Tuple[Dict[str, str], str]]):
        class_counts: Dict[str, int] = defaultdict(int)
        feature_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for features, label in data:
            self.classes.add(label)
            class_counts[label] += 1
            for feat_name, feat_value in features.items():
                self.features.add(feat_name)
                feature_counts[label][feat_name][feat_value] += 1
        
        total = sum(class_counts.values())
        self.class_prior = {c: count / total for c, count in class_counts.items()}
        
        for label in self.classes:
            self.feature_probs[label] = {}
            for feat_name in self.features:
                values = feature_counts[label][feat_name]
                total_feat = sum(values.values())
                denom = total_feat + self.smoothing * max(1, len(values))
                self.feature_probs[label][feat_name] = {
                    v: (count + self.smoothing) / denom
                    for v, count in values.items()
                }
    
    def predict(self, features: Dict[str, str]) -> str:
        scores = {}
        for label in self.classes:
            log_prob = math.log(self.class_prior[label])
            for feat_name, feat_value in features.items():
                prob = self.feature_probs[label][feat_name].get(feat_value, self.smoothing)
                log_prob += math.log(prob)
            scores[label] = log_prob
        
        return max(scores, key=scores.get)
    
    def predict_proba(self, features: Dict[str, str]) -> Dict[str, float]:
        scores = {}
        for label in self.classes:
            log_prob = math.log(self.class_prior[label])
            for feat_name, feat_value in features.items():
                prob = self.feature_probs[label][feat_name].get(feat_value, self.smoothing)
                log_prob += math.log(prob)
            scores[label] = log_prob
        
        max_log = max(scores.values())
        probs = {label: math.exp(score - max_log) for label, score in scores.items()}
        total = sum(probs.values())
        return {label: p / total for label, p in probs.items()}


class BayesianNetwork:
    """
    贝叶斯网络
    
    基于 Pearl (1988) 的概率推理方法
    使用有向无环图表示变量依赖关系
    """
    
    def __init__(self):
        self.nodes: Dict[str, BayesianNode] = {}
        self.graph: Dict[str, List[str]] = {}
    
    def add_node(self, name: str, parents: List[str], cpt: Dict[Tuple[str, ...], float]):
        node = BayesianNode(name=name, parents=parents, cpt=cpt)
        self.nodes[name] = node
        
        for parent in parents:
            if parent not in self.graph:
                self.graph[parent] = []
            self.graph[parent].append(name)
        
        if name not in self.graph:
            self.graph[name] = []
    
    def get_probability(self, node_name: str, value: str, evidence: Dict[str, str]) -> float:
        node = self.nodes[node_name]
        
        if not node.parents:
            return node.cpt.get((value,), 0.5)
        
        parent_values = tuple(evidence.get(p, '') for p in node.parents)
        return node.cpt.get(parent_values + (value,), 0.5)
    
    def compute_joint(self, assignment: Dict[str, str]) -> float:
        prob = 1.0
        for node_name, value in assignment.items():
            node = self.nodes[node_name]
            parent_values = tuple(assignment.get(p, '') for p in node.parents)
            key = parent_values + (value,) if node.parents else (value,)
            prob *= node.cpt.get(key, 0.5)
        return prob
    
    def inference_by_enumeration(self, query: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        枚举推理算法
        
        基于 Pearl (1988) 的精确推理方法
        """
        hidden_vars = [n for n in self.nodes if n not in evidence and n != query]
        
        results = {}
        query_node = self.nodes[query]
        
        possible_values = set()
        for key in query_node.cpt.keys():
            possible_values.add(key[-1])
        
        for query_value in possible_values:
            total_prob = 0.0
            
            for hidden_assignment in self._enumerate_assignments(hidden_vars):
                full_assignment = {**evidence, **hidden_assignment, query: query_value}
                prob = self.compute_joint(full_assignment)
                total_prob += prob
            
            results[query_value] = total_prob
        
        total = sum(results.values())
        if total > 0:
            return {v: p / total for v, p in results.items()}
        return results
    
    def _enumerate_assignments(self, vars: List[str]) -> List[Dict[str, str]]:
        if not vars:
            return [{}]
        
        result = []
        remaining = vars[1:]
        
        node = self.nodes[vars[0]]
        possible_values = set()
        for key in node.cpt.keys():
            possible_values.add(key[-1])
        
        for value in possible_values:
            for assignment in self._enumerate_assignments(remaining):
                result.append({vars[0]: value, **assignment})
        
        return result


class BayesianInferenceEngine:
    """
    贝叶斯推理引擎
    
    实现 Pearl (1988) 的贝叶斯网络推理
    """
    
    def __init__(self, network: BayesianNetwork):
        self.network = network
    
    def update_beliefs(self, evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        beliefs = {}
        for node_name in self.network.nodes:
            beliefs[node_name] = self.network.inference_by_enumeration(node_name, evidence)
        return beliefs
    
    def most_probable_explanation(self, evidence: Dict[str, str]) -> Dict[str, str]:
        """
        最可能解释 (MPE)
        
        找到给定证据下最可能的完整赋值
        """
        hidden_vars = [n for n in self.network.nodes if n not in evidence]
        
        best_assignment = None
        best_prob = -1
        
        for assignment in self.network._enumerate_assignments(hidden_vars):
            full_assignment = {**evidence, **assignment}
            prob = self.network.compute_joint(full_assignment)
            if prob > best_prob:
                best_prob = prob
                best_assignment = full_assignment
        
        return best_assignment or {}


def create_medical_diagnosis_network() -> BayesianNetwork:
    """
    创建医疗诊断贝叶斯网络示例
    
    基于 Heckerman (1995) 的医疗诊断示例
    """
    network = BayesianNetwork()
    
    network.add_node('smoking', [], {
        ('yes',): 0.3,
        ('no',): 0.7
    })
    
    network.add_node('lung_disease', ['smoking'], {
        ('yes', 'present'): 0.1,
        ('yes', 'absent'): 0.9,
        ('no', 'present'): 0.01,
        ('no', 'absent'): 0.99
    })
    
    network.add_node('cough', ['lung_disease'], {
        ('present', 'yes'): 0.8,
        ('present', 'no'): 0.2,
        ('absent', 'yes'): 0.1,
        ('absent', 'no'): 0.9
    })
    
    network.add_node('xray', ['lung_disease'], {
        ('present', 'abnormal'): 0.9,
        ('present', 'normal'): 0.1,
        ('absent', 'abnormal'): 0.05,
        ('absent', 'normal'): 0.95
    })
    
    return network


if __name__ == "__main__":
    print("="*60)
    print("朴素贝叶斯分类器测试")
    print("="*60)
    
    nb = NaiveBayesClassifier()
    
    training_data = [
        ({'color': 'red', 'shape': 'round', 'size': 'medium'}, 'apple'),
        ({'color': 'red', 'shape': 'round', 'size': 'small'}, 'apple'),
        ({'color': 'yellow', 'shape': 'long', 'size': 'medium'}, 'banana'),
        ({'color': 'yellow', 'shape': 'long', 'size': 'large'}, 'banana'),
        ({'color': 'orange', 'shape': 'round', 'size': 'medium'}, 'orange'),
        ({'color': 'green', 'shape': 'round', 'size': 'small'}, 'grape'),
    ]
    
    nb.fit(training_data)
    
    test_features = {'color': 'red', 'shape': 'round', 'size': 'large'}
    prediction = nb.predict(test_features)
    probs = nb.predict_proba(test_features)
    
    print(f"\n测试特征: {test_features}")
    print(f"预测类别: {prediction}")
    print(f"概率分布: {probs}")
    
    print("\n" + "="*60)
    print("贝叶斯网络测试 - 医疗诊断")
    print("="*60)
    
    network = create_medical_diagnosis_network()
    engine = BayesianInferenceEngine(network)
    
    evidence1 = {'cough': 'yes'}
    beliefs1 = engine.update_beliefs(evidence1)
    
    print(f"\n证据: 咳嗽=yes")
    print(f"肺癌概率: {beliefs1['lung_disease']}")
    print(f"吸烟概率: {beliefs1['smoking']}")
    
    evidence2 = {'cough': 'yes', 'xray': 'abnormal'}
    beliefs2 = engine.update_beliefs(evidence2)
    
    print(f"\n证据: 咳嗽=yes, X光异常")
    print(f"肺癌概率: {beliefs2['lung_disease']}")
    
    mpe = engine.most_probable_explanation(evidence2)
    print(f"\n最可能解释: {mpe}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Pearl (1988) - 贝叶斯网络奠基著作")
    print("Heckerman (1995) - 贝叶斯网络学习教程")
    print("Maron (1961) - 朴素贝叶斯分类器起源")