#!/usr/bin/env python3
"""
k-近邻算法实现

参考文献:
@article{fix1951,
  title={Discriminatory analysis. Nonparametric discrimination: Consistency properties},
  author={Fix, Evelyn and Hodges, Joseph L.},
  publisher={USA School of Aviation Medicine},
  year={1951}
}

@article{cover1967,
  title={Nearest neighbor pattern classification},
  author={Cover, Thomas M. and Hart, Peter E.},
  journal={IEEE Transactions on Information Theory},
  volume={13},
  number={1},
  pages={21--27},
  year={1967}
}

@book{duda2000,
  title={Pattern Classification},
  author={Duda, Richard O. and Hart, Peter E. and Stork, David G.},
  publisher={Wiley},
  year={2000}
}
"""

from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from collections import Counter
import math


@dataclass
class Neighbor:
    features: List[float]
    label: str
    distance: float


class DistanceMetric:
    """
    距离度量
    
    基于 Cover & Hart (1967) 的距离度量方法
    """
    
    @staticmethod
    def euclidean(x1: List[float], x2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    @staticmethod
    def manhattan(x1: List[float], x2: List[float]) -> float:
        return sum(abs(a - b) for a, b in zip(x1, x2))
    
    @staticmethod
    def minkowski(x1: List[float], x2: List[float], p: int = 3) -> float:
        return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1 / p)
    
    @staticmethod
    def cosine(x1: List[float], x2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(x1, x2))
        norm1 = math.sqrt(sum(a ** 2 for a in x1))
        norm2 = math.sqrt(sum(b ** 2 for b in x2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return 1.0 - dot / (norm1 * norm2)
    
    @staticmethod
    def hamming(x1: List, x2: List) -> float:
        return sum(1 for a, b in zip(x1, x2) if a != b) / len(x1)


class KNNClassifier:
    """
    k-近邻分类器
    
    基于 Cover & Hart (1967) 的k-NN算法
    使用多数投票进行分类
    """
    
    def __init__(self, k: int = 3, 
                 distance_metric: Callable[[List[float], List[float]], float] = DistanceMetric.euclidean,
                 weighted: bool = False):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.X_train: List[List[float]] = []
        self.y_train: List[str] = []
    
    def fit(self, X: List[List[float]], y: List[str]):
        self.X_train = X
        self.y_train = y
    
    def _find_neighbors(self, x: List[float]) -> List[Neighbor]:
        distances = []
        for xi, yi in zip(self.X_train, self.y_train):
            dist = self.distance_metric(x, xi)
            distances.append(Neighbor(features=xi, label=yi, distance=dist))
        
        distances.sort(key=lambda n: n.distance)
        return distances[:self.k]
    
    def predict(self, x: List[float]) -> str:
        neighbors = self._find_neighbors(x)
        
        if self.weighted:
            votes: Dict[str, float] = {}
            for neighbor in neighbors:
                weight = 1.0 / (neighbor.distance + 1e-10)
                votes[neighbor.label] = votes.get(neighbor.label, 0.0) + weight
            return max(votes, key=votes.get)
        else:
            labels = [n.label for n in neighbors]
            return Counter(labels).most_common(1)[0][0]
    
    def predict_batch(self, X: List[List[float]]) -> List[str]:
        return [self.predict(x) for x in X]
    
    def predict_proba(self, x: List[float]) -> Dict[str, float]:
        neighbors = self._find_neighbors(x)
        
        if self.weighted:
            votes: Dict[str, float] = {}
            for neighbor in neighbors:
                weight = 1.0 / (neighbor.distance + 1e-10)
                votes[neighbor.label] = votes.get(neighbor.label, 0.0) + weight
            total = sum(votes.values())
            return {label: weight / total for label, weight in votes.items()}
        else:
            labels = [n.label for n in neighbors]
            counts = Counter(labels)
            total = sum(counts.values())
            return {label: count / total for label, count in counts.items()}


class KNNRegressor:
    """
    k-近邻回归
    
    使用k个最近邻的平均值进行回归预测
    """
    
    def __init__(self, k: int = 3,
                 distance_metric: Callable[[List[float], List[float]], float] = DistanceMetric.euclidean,
                 weighted: bool = False):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[float]):
        self.X_train = X
        self.y_train = y
    
    def predict(self, x: List[float]) -> float:
        distances = []
        for xi, yi in zip(self.X_train, self.y_train):
            dist = self.distance_metric(x, xi)
            distances.append((yi, dist))
        
        distances.sort(key=lambda t: t[1])
        nearest = distances[:self.k]
        
        if self.weighted:
            weighted_sum = 0.0
            weight_sum = 0.0
            for yi, dist in nearest:
                weight = 1.0 / (dist + 1e-10)
                weighted_sum += yi * weight
                weight_sum += weight
            return weighted_sum / weight_sum
        else:
            return sum(yi for yi, _ in nearest) / len(nearest)
    
    def predict_batch(self, X: List[List[float]]) -> List[float]:
        return [self.predict(x) for x in X]


class KDTreeKNN:
    """
    使用KD-Tree加速的k-NN
    
    KD-Tree是一种空间划分数据结构
    可将搜索复杂度从O(n)降至O(log n)
    """
    
    def __init__(self, k: int = 3):
        self.k = k
        self.root: Optional['KDNode'] = None
    
    def fit(self, X: List[List[float]], y: List[str]):
        points = [(x, y) for x, y in zip(X, y)]
        self.root = self._build_tree(points, depth=0)
    
    def _build_tree(self, points: List[Tuple[List[float], str]], depth: int) -> Optional['KDNode']:
        if not points:
            return None
        
        n_features = len(points[0][0])
        axis = depth % n_features
        
        points.sort(key=lambda p: p[0][axis])
        median_idx = len(points) // 2
        
        return KDNode(
            point=points[median_idx][0],
            label=points[median_idx][1],
            axis=axis,
            left=self._build_tree(points[:median_idx], depth + 1),
            right=self._build_tree(points[median_idx + 1:], depth + 1)
        )
    
    def predict(self, x: List[float]) -> str:
        neighbors = self._search(x, self.root, [])
        
        labels = [n[1] for n in neighbors[:self.k]]
        return Counter(labels).most_common(1)[0][0]
    
    def _search(self, x: List[float], node: Optional['KDNode'], 
               best: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        if node is None:
            return best
        
        dist = DistanceMetric.euclidean(x, node.point)
        
        if len(best) < self.k:
            best.append((dist, node.label))
            best.sort(key=lambda t: t[0])
        elif dist < best[-1][0]:
            best[-1] = (dist, node.label)
            best.sort(key=lambda t: t[0])
        
        diff = x[node.axis] - node.point[node.axis]
        
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        
        best = self._search(x, near, best)
        
        if diff ** 2 < best[-1][0] ** 2 or len(best) < self.k:
            best = self._search(x, far, best)
        
        return best


@dataclass
class KDNode:
    point: List[float]
    label: str
    axis: int
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None


class BallTreeKNN:
    """
    Ball Tree k-NN
    
    使用球树进行空间划分
    适用于高维数据
    """
    
    def __init__(self, k: int = 3):
        self.k = k
        self.root: Optional['BallNode'] = None
    
    def fit(self, X: List[List[float]], y: List[str]):
        points = [(x, y) for x, y in zip(X, y)]
        self.root = self._build_ball(points)
    
    def _build_ball(self, points: List[Tuple[List[float], str]]) -> Optional['BallNode']:
        if not points:
            return None
        
        centroid = self._compute_centroid([p[0] for p in points])
        radius = max(DistanceMetric.euclidean(centroid, p[0]) for p in points)
        
        if len(points) <= 2:
            return BallNode(centroid=centroid, radius=radius, points=points)
        
        distances = [(DistanceMetric.euclidean(centroid, p[0]), p) for p in points]
        distances.sort(key=lambda t: t[0])
        
        median_idx = len(distances) // 2
        left_points = [d[1] for d in distances[:median_idx]]
        right_points = [d[1] for d in distances[median_idx:]]
        
        return BallNode(
            centroid=centroid,
            radius=radius,
            left=self._build_ball(left_points),
            right=self._build_ball(right_points)
        )
    
    def _compute_centroid(self, points: List[List[float]]) -> List[float]:
        if not points:
            return []
        n = len(points)
        n_features = len(points[0])
        centroid = [sum(p[i] for p in points) / n for i in range(n_features)]
        return centroid
    
    def predict(self, x: List[float]) -> str:
        neighbors = self._search_ball(x, self.root, [])
        labels = [n[1] for n in neighbors[:self.k]]
        return Counter(labels).most_common(1)[0][0]
    
    def _search_ball(self, x: List[float], node: Optional['BallNode'],
                    best: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        if node is None:
            return best
        
        if node.points:
            for point, label in node.points:
                dist = DistanceMetric.euclidean(x, point)
                if len(best) < self.k:
                    best.append((dist, label))
                    best.sort(key=lambda t: t[0])
                elif dist < best[-1][0]:
                    best[-1] = (dist, label)
                    best.sort(key=lambda t: t[0])
            return best
        
        dist_to_centroid = DistanceMetric.euclidean(x, node.centroid)
        
        if dist_to_centroid - node.radius > best[-1][0] if best else float('inf'):
            return best
        
        best = self._search_ball(x, node.left, best)
        best = self._search_ball(x, node.right, best)
        
        return best


@dataclass
class BallNode:
    centroid: List[float]
    radius: float
    points: List[Tuple[List[float], str]] = None
    left: Optional['BallNode'] = None
    right: Optional['BallNode'] = None


if __name__ == "__main__":
    print("="*60)
    print("k-NN分类器测试")
    print("="*60)
    
    knn = KNNClassifier(k=3, weighted=True)
    
    X_train = [
        [1.0, 1.0], [1.5, 1.5], [2.0, 2.0],
        [5.0, 5.0], [5.5, 5.5], [6.0, 6.0],
        [3.0, 1.0], [1.0, 3.0]
    ]
    y_train = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C']
    
    knn.fit(X_train, y_train)
    
    print(f"\n训练数据: {len(X_train)} 个样本")
    
    test_points = [
        [1.2, 1.2],
        [5.2, 5.2],
        [2.0, 2.5],
        [4.0, 4.0]
    ]
    
    for point in test_points:
        pred = knn.predict(point)
        probs = knn.predict_proba(point)
        print(f"\n测试点 {point}: 预测={pred}")
        print(f"  概率分布: {probs}")
    
    print("\n" + "="*60)
    print("距离度量测试")
    print("="*60)
    
    p1 = [1.0, 2.0, 3.0]
    p2 = [4.0, 5.0, 6.0]
    
    print(f"\n点 p1={p1}, p2={p2}")
    print(f"欧氏距离: {DistanceMetric.euclidean(p1, p2):.4f}")
    print(f"曼哈顿距离: {DistanceMetric.manhattan(p1, p2):.4f}")
    print(f"Minkowski距离(p=3): {DistanceMetric.minkowski(p1, p2, p=3):.4f}")
    print(f"余弦距离: {DistanceMetric.cosine(p1, p2):.4f}")
    
    print("\n" + "="*60)
    print("KD-Tree k-NN测试")
    print("="*60)
    
    kd_knn = KDTreeKNN(k=3)
    kd_knn.fit(X_train, y_train)
    
    kd_pred = kd_knn.predict([1.2, 1.2])
    print(f"\nKD-Tree预测 [1.2, 1.2]: {kd_pred}")
    
    print("\n" + "="*60)
    print("k-NN回归测试")
    print("="*60)
    
    knn_reg = KNNRegressor(k=3, weighted=True)
    
    X_reg = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y_reg = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    knn_reg.fit(X_reg, y_reg)
    
    test_values = [[1.5], [2.5], [3.5]]
    predictions = knn_reg.predict_batch(test_values)
    
    print(f"\n训练数据: x=[1,2,3,4,5], y=[2,4,6,8,10]")
    for x, y_pred in zip(test_values, predictions):
        print(f"测试 {x[0]}: 预测={y_pred:.2f}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Fix & Hodges (1951) - k-NN起源论文")
    print("Cover & Hart (1967) - k-NN理论分析")
    print("Duda, Hart & Stork (2000) - 模式识别经典教材")