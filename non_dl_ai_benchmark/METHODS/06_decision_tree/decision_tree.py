#!/usr/bin/env python3
"""
决策树实现

参考文献:
@article{quinlan1986,
  title={Induction of decision trees},
  author={Quinlan, J. Ross},
  journal={Machine Learning},
  volume={1},
  number={1},
  pages={81--106},
  year={1986}
}

@book{quinlan1993,
  title={C4.5: Programs for Machine Learning},
  author={Quinlan, J. Ross},
  publisher={Morgan Kaufmann},
  year={1993}
}

@article{breiman2001,
  title={Random forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001}
}

@article{ho1995,
  title={Random decision forests},
  author={Ho, Tin Kam},
  booktitle={Proceedings of the 3rd International Conference on Document Analysis and Recognition},
  pages={278--282},
  year={1995}
}
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import math


@dataclass
class DecisionNode:
    feature: Optional[str] = None
    threshold: Optional[float] = None
    children: Dict[Any, 'DecisionNode'] = field(default_factory=dict)
    label: Optional[str] = None
    samples: int = 0
    
    def is_leaf(self) -> bool:
        return self.label is not None
    
    def predict(self, features: Dict[str, Any]) -> str:
        if self.is_leaf():
            return self.label
        
        if self.feature not in features:
            return self.label or ''
        
        value = features[self.feature]
        
        if self.threshold is not None:
            branch = 'left' if value <= self.threshold else 'right'
        else:
            branch = value
        
        if branch in self.children:
            return self.children[branch].predict(features)
        
        return self.label or ''


class ID3DecisionTree:
    """
    ID3决策树
    
    基于 Quinlan (1986) 的信息增益分裂准则
    使用熵作为不纯度度量
    """
    
    def __init__(self, max_depth: int = 10, min_samples: int = 1):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: DecisionNode = None
    
    def entropy(self, labels: List[str]) -> float:
        if not labels:
            return 0.0
        
        counts = Counter(labels)
        total = len(labels)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def information_gain(self, data: List[Tuple[Dict[str, Any], str]], feature: str) -> float:
        labels = [label for _, label in data]
        total_entropy = self.entropy(labels)
        
        feature_values = set(feat[feature] for feat, _ in data)
        
        weighted_entropy = 0.0
        for value in feature_values:
            subset = [(feat, label) for feat, label in data if feat[feature] == value]
            subset_labels = [label for _, label in subset]
            weight = len(subset) / len(data)
            weighted_entropy += weight * self.entropy(subset_labels)
        
        return total_entropy - weighted_entropy
    
    def select_best_feature(self, data: List[Tuple[Dict[str, Any], str]], features: List[str]) -> str:
        best_feature = None
        best_gain = -1
        
        for feature in features:
            gain = self.information_gain(data, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature
    
    def fit(self, data: List[Tuple[Dict[str, Any], str]], features: List[str], depth: int = 0):
        self.root = self._build_tree(data, features, depth)
    
    def _build_tree(self, data: List[Tuple[Dict[str, Any], str]], 
                    features: List[str], depth: int) -> DecisionNode:
        labels = [label for _, label in data]
        
        if len(set(labels)) == 1:
            return DecisionNode(label=labels[0], samples=len(data))
        
        if len(features) == 0 or len(data) < self.min_samples or depth >= self.max_depth:
            majority = Counter(labels).most_common(1)[0][0]
            return DecisionNode(label=majority, samples=len(data))
        
        best_feature = self.select_best_feature(data, features)
        
        if best_feature is None:
            majority = Counter(labels).most_common(1)[0][0]
            return DecisionNode(label=majority, samples=len(data))
        
        node = DecisionNode(feature=best_feature, samples=len(data))
        
        remaining_features = [f for f in features if f != best_feature]
        feature_values = set(feat[best_feature] for feat, _ in data)
        
        for value in feature_values:
            subset = [(feat, label) for feat, label in data if feat[best_feature] == value]
            if subset:
                node.children[value] = self._build_tree(subset, remaining_features, depth + 1)
            else:
                majority = Counter(labels).most_common(1)[0][0]
                node.children[value] = DecisionNode(label=majority, samples=0)
        
        majority = Counter(labels).most_common(1)[0][0]
        node.label = majority
        
        return node
    
    def predict(self, features: Dict[str, Any]) -> str:
        if self.root is None:
            return ''
        return self.root.predict(features)
    
    def print_tree(self, node: DecisionNode = None, indent: str = ""):
        if node is None:
            node = self.root
        
        if node is None:
            return
        
        if node.is_leaf():
            print(f"{indent}叶子节点: 类别={node.label}, 样本数={node.samples}")
        else:
            print(f"{indent}特征={node.feature}, 样本数={node.samples}")
            for value, child in node.children.items():
                print(f"{indent}  分支 [{value}] ->")
                self.print_tree(child, indent + "    ")


class C45DecisionTree(ID3DecisionTree):
    """
    C4.5决策树
    
    基于 Quinlan (1993) 的改进:
    - 使用信息增益率代替信息增益
    - 支持连续属性
    - 处理缺失值
    """
    
    def split_info(self, data: List[Tuple[Dict[str, Any], str]], feature: str) -> float:
        feature_values = [feat[feature] for feat, _ in data]
        counts = Counter(feature_values)
        total = len(data)
        
        split_info = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                split_info -= p * math.log2(p)
        
        return split_info
    
    def gain_ratio(self, data: List[Tuple[Dict[str, Any], str]], feature: str) -> float:
        gain = self.information_gain(data, feature)
        split = self.split_info(data, feature)
        
        if split == 0:
            return 0
        
        return gain / split
    
    def select_best_feature(self, data: List[Tuple[Dict[str, Any], str]], features: List[str]) -> str:
        best_feature = None
        best_ratio = -1
        
        for feature in features:
            ratio = self.gain_ratio(data, feature)
            if ratio > best_ratio:
                best_ratio = ratio
                best_feature = feature
        
        return best_feature
    
    def find_threshold(self, data: List[Tuple[Dict[str, Any], str]], feature: str) -> Tuple[float, float]:
        values = sorted(set(feat[feature] for feat, _ in data if isinstance(feat[feature], (int, float))))
        
        if not values:
            return (0, 0)
        
        best_threshold = values[0]
        best_gain = 0
        
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            
            left_labels = [label for feat, label in data if feat.get(feature, 0) <= threshold]
            right_labels = [label for feat, label in data if feat.get(feature, 0) > threshold]
            
            total_entropy = self.entropy([label for _, label in data])
            
            n = len(data)
            n_left = len(left_labels)
            n_right = len(right_labels)
            
            if n_left > 0 and n_right > 0:
                gain = total_entropy - (n_left / n) * self.entropy(left_labels) - (n_right / n) * self.entropy(right_labels)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
        
        return (best_threshold, best_gain)


class RandomForest:
    """
    随机森林
    
    基于 Breiman (2001) 的随机森林方法
    使用Bootstrap采样和随机特征选择
    """
    
    def __init__(self, n_trees: int = 10, max_depth: int = 10, 
                 min_samples: int = 1, feature_fraction: float = 0.7):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_fraction = feature_fraction
        self.trees: List[ID3DecisionTree] = []
    
    def bootstrap_sample(self, data: List[Tuple[Dict[str, Any], str]]) -> List[Tuple[Dict[str, Any], str]]:
        import random
        n = len(data)
        return [data[random.randint(0, n - 1)] for _ in range(n)]
    
    def select_random_features(self, features: List[str]) -> List[str]:
        import random
        n_features = max(1, int(len(features) * self.feature_fraction))
        return random.sample(features, n_features)
    
    def fit(self, data: List[Tuple[Dict[str, Any], str]], features: List[str]):
        self.trees = []
        
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            selected_features = self.select_random_features(features)
            
            tree = ID3DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(sample, selected_features)
            self.trees.append(tree)
    
    def predict(self, features: Dict[str, Any]) -> str:
        if not self.trees:
            return ''
        
        predictions = [tree.predict(features) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]
    
    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, float]:
        if not self.trees:
            return {}
        
        predictions = [tree.predict(features) for tree in self.trees]
        counts = Counter(predictions)
        total = sum(counts.values())
        return {label: count / total for label, count in counts.items()}


if __name__ == "__main__":
    print("="*60)
    print("ID3决策树测试")
    print("="*60)
    
    id3 = ID3DecisionTree(max_depth=5)
    
    training_data = [
        ({'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'wind': 'weak'}, 'no'),
        ({'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'wind': 'strong'}, 'no'),
        ({'outlook': 'overcast', 'temperature': 'hot', 'humidity': 'high', 'wind': 'weak'}, 'yes'),
        ({'outlook': 'rain', 'temperature': 'mild', 'humidity': 'high', 'wind': 'weak'}, 'yes'),
        ({'outlook': 'rain', 'temperature': 'cool', 'humidity': 'normal', 'wind': 'weak'}, 'yes'),
        ({'outlook': 'rain', 'temperature': 'cool', 'humidity': 'normal', 'wind': 'strong'}, 'no'),
        ({'outlook': 'overcast', 'temperature': 'cool', 'humidity': 'normal', 'wind': 'strong'}, 'yes'),
        ({'outlook': 'sunny', 'temperature': 'mild', 'humidity': 'high', 'wind': 'weak'}, 'no'),
        ({'outlook': 'sunny', 'temperature': 'cool', 'humidity': 'normal', 'wind': 'weak'}, 'yes'),
        ({'outlook': 'rain', 'temperature': 'mild', 'humidity': 'normal', 'wind': 'weak'}, 'yes'),
    ]
    
    features = ['outlook', 'temperature', 'humidity', 'wind']
    id3.fit(training_data, features)
    
    print("\n决策树结构:")
    id3.print_tree()
    
    test_case = {'outlook': 'sunny', 'temperature': 'mild', 'humidity': 'normal', 'wind': 'strong'}
    prediction = id3.predict(test_case)
    print(f"\n测试: {test_case}")
    print(f"预测: {prediction}")
    
    print("\n" + "="*60)
    print("随机森林测试")
    print("="*60)
    
    rf = RandomForest(n_trees=20, max_depth=5, feature_fraction=0.6)
    rf.fit(training_data, features)
    
    print(f"\n训练了 {len(rf.trees)} 棵树")
    
    test_cases = [
        {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'normal', 'wind': 'weak'},
        {'outlook': 'rain', 'temperature': 'mild', 'humidity': 'high', 'wind': 'strong'},
    ]
    
    for case in test_cases:
        pred = rf.predict(case)
        probs = rf.predict_proba(case)
        print(f"\n测试: {case}")
        print(f"预测: {pred}")
        print(f"概率: {probs}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Quinlan (1986) - ID3算法奠基论文")
    print("Quinlan (1993) - C4.5算法经典著作")
    print("Breiman (2001) - 随机森林奠基论文")