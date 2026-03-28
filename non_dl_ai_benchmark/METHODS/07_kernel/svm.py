#!/usr/bin/env python3
"""
支持向量机实现

参考文献:
@article{cortes1995,
  title={Support-vector networks},
  author={Cortes, Corinna and Vapnik, Vladimir},
  journal={Machine Learning},
  volume={20},
  number={3},
  pages={273--297},
  year={1995}
}

@book{vapnik1995,
  title={The Nature of Statistical Learning Theory},
  author={Vapnik, Vladimir N.},
  publisher={Springer},
  year={1995}
}

@article{burges1998,
  title={A tutorial on support vector machines for pattern recognition},
  author={Burges, Christopher J. C.},
  journal={Data Mining and Knowledge Discovery},
  volume={2},
  number={2},
  pages={121--167},
  year={1998}
}

@article{joachims1998,
  title={Text categorization with support vector machines: Learning with many relevant features},
  author={Joachims, Thorsten},
  booktitle={Machine Learning: ECML-98},
  pages={137--142},
  year={1998}
}
"""

from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import math


@dataclass
class SVMParameters:
    weights: List[float]
    bias: float
    support_vectors: List[Tuple[List[float], float]]
    kernel_type: str
    C: float


class Kernel:
    """
    核函数
    
    基于 Vapnik (1995) 的核方法
    将低维数据映射到高维空间
    """
    
    @staticmethod
    def linear(x1: List[float], x2: List[float]) -> float:
        return sum(a * b for a, b in zip(x1, x2))
    
    @staticmethod
    def polynomial(x1: List[float], x2: List[float], degree: int = 2, coef0: float = 1.0) -> float:
        dot = sum(a * b for a, b in zip(x1, x2))
        return (coef0 + dot) ** degree
    
    @staticmethod
    def rbf(x1: List[float], x2: List[float], gamma: float = 0.1) -> float:
        diff_sq = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-gamma * diff_sq)
    
    @staticmethod
    def sigmoid(x1: List[float], x2: List[float], gamma: float = 0.1, coef0: float = 0.0) -> float:
        dot = sum(a * b for a, b in zip(x1, x2))
        return math.tanh(gamma * dot + coef0)


class LinearSVM:
    """
    线性支持向量机
    
    基于 Cortes & Vapnik (1995) 的软间隔SVM
    使用简化版本的SMO算法
    """
    
    def __init__(self, C: float = 1.0, max_iterations: int = 1000, 
                 learning_rate: float = 0.01, tolerance: float = 1e-4):
        self.C = C
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.support_vectors: List[Tuple[List[float], float]] = []
    
    def fit(self, X: List[List[float]], y: List[float]):
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            errors = []
            
            for i in range(n_samples):
                prediction = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                error = y[i] * prediction
                
                errors.append(prediction - y[i])
                
                if error < 1:
                    for j in range(n_features):
                        self.weights[j] += self.learning_rate * (
                            y[i] * X[i][j] - 2 * self.C * self.weights[j]
                        )
                    self.bias += self.learning_rate * y[i]
            
            total_error = sum(abs(e) for e in errors) / n_samples
            if total_error < self.tolerance:
                break
        
        self._identify_support_vectors(X, y)
    
    def _identify_support_vectors(self, X: List[List[float]], y: List[float]):
        self.support_vectors = []
        for i, (xi, yi) in enumerate(zip(X, y)):
            prediction = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
            margin = yi * prediction
            
            if margin <= 1 + self.tolerance:
                self.support_vectors.append((xi, yi))
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for xi in X:
            score = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
            predictions.append(1.0 if score >= 0 else -1.0)
        return predictions
    
    def decision_function(self, x: List[float]) -> float:
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
    
    def margin(self, x: List[float], y: float) -> float:
        return y * self.decision_function(x)


class KernelSVM:
    """
    核支持向量机
    
    基于 Vapnik (1995) 的核方法实现非线性分类
    使用简化版本的SMO算法
    """
    
    def __init__(self, kernel: Callable[[List[float], List[float]], float] = Kernel.rbf,
                 C: float = 1.0, max_iterations: int = 100, tolerance: float = 1e-3):
        self.kernel = kernel
        self.C = C
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alphas: List[float] = []
        self.bias: float = 0.0
        self.support_vectors: List[Tuple[List[float], float]] = []
        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[float]):
        n_samples = len(X)
        
        self.X_train = X
        self.y_train = y
        self.alphas = [0.0] * n_samples
        self.bias = 0.0
        
        kernel_matrix = self._compute_kernel_matrix(X)
        
        for iteration in range(self.max_iterations):
            for i in range(n_samples):
                Ei = self._decision_function_single(i) - y[i]
                
                if (y[i] * Ei < -self.tolerance and self.alphas[i] < self.C) or \
                   (y[i] * Ei > self.tolerance and self.alphas[i] > 0):
                    
                    j = (i + 1) % n_samples
                    while j == i:
                        j = (j + 1) % n_samples
                    
                    Ej = self._decision_function_single(j) - y[j]
                    
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j]
                    
                    if eta >= 0:
                        continue
                    
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = max(L, min(H, self.alphas[j]))
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    b1 = self.bias - Ei - y[i] * (self.alphas[i] - alpha_i_old) * kernel_matrix[i][i] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * kernel_matrix[i][j]
                    
                    b2 = self.bias - Ej - y[i] * (self.alphas[i] - alpha_i_old) * kernel_matrix[i][j] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * kernel_matrix[j][j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.bias = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.bias = b2
                    else:
                        self.bias = (b1 + b2) / 2
        
        self._extract_support_vectors()
    
    def _compute_kernel_matrix(self, X: List[List[float]]) -> List[List[float]]:
        n = len(X)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.kernel(X[i], X[j])
        return matrix
    
    def _decision_function_single(self, i: int) -> float:
        result = 0.0
        for j in range(len(self.X_train)):
            result += self.alphas[j] * self.y_train[j] * self.kernel(self.X_train[j], self.X_train[i])
        return result + self.bias
    
    def _extract_support_vectors(self):
        self.support_vectors = []
        for i in range(len(self.alphas)):
            if self.alphas[i] > 1e-5:
                self.support_vectors.append((self.X_train[i], self.y_train[i]))
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for xi in X:
            score = 0.0
            for j, (sv, sv_y) in enumerate(self.support_vectors):
                alpha_idx = [k for k in range(len(self.X_train)) if self.X_train[k] == sv][0]
                alpha = self.alphas[alpha_idx]
                score += alpha * sv_y * self.kernel(sv, xi)
            score += self.bias
            predictions.append(1.0 if score >= 0 else -1.0)
        return predictions
    
    def decision_function(self, x: List[float]) -> float:
        score = 0.0
        for j, (sv, sv_y) in enumerate(self.support_vectors):
            alpha_idx = [k for k in range(len(self.X_train)) if self.X_train[k] == sv][0]
            alpha = self.alphas[alpha_idx]
            score += alpha * sv_y * self.kernel(sv, x)
        return score + self.bias


class SVR:
    """
    支持向量回归
    
    基于 Vapnik (1995) 的ε-SVR算法
    """
    
    def __init__(self, kernel: Callable[[List[float], List[float]], float] = Kernel.rbf,
                 C: float = 1.0, epsilon: float = 0.1, max_iterations: int = 100):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.alphas: List[float] = []
        self.bias: float = 0.0
        self.support_vectors: List[List[float]] = []
        self.y_train: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[float]):
        n = len(X)
        self.alphas = [0.0] * n
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            for i in range(n):
                pred = self._predict_single(X[i])
                error = pred - y[i]
                
                if abs(error) > self.epsilon:
                    update = min(self.C, abs(error) - self.epsilon)
                    self.alphas[i] += update if error > 0 else -update
        
        self.support_vectors = [X[i] for i in range(n) if abs(self.alphas[i]) > 1e-5]
        self.y_train = y
    
    def _predict_single(self, x: List[float]) -> float:
        result = self.bias
        for i, alpha in enumerate(self.alphas):
            if abs(alpha) > 1e-5:
                result += alpha * self.kernel(x, x)
        return result
    
    def predict(self, X: List[List[float]]) -> List[float]:
        return [self._predict_single(xi) for xi in X]


if __name__ == "__main__":
    print("="*60)
    print("线性SVM测试")
    print("="*60)
    
    svm = LinearSVM(C=1.0, max_iterations=1000, learning_rate=0.01)
    
    X = [
        [1.0, 1.0], [2.0, 1.0], [1.0, 2.0],
        [4.0, 4.0], [5.0, 4.0], [4.0, 5.0]
    ]
    y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    
    svm.fit(X, y)
    
    print(f"\n权重: {svm.weights}")
    print(f"偏置: {svm.bias}")
    print(f"支持向量数量: {len(svm.support_vectors)}")
    
    predictions = svm.predict(X)
    accuracy = sum(1 for p, yt in zip(predictions, y) if p == yt) / len(y)
    print(f"训练准确率: {accuracy:.2%}")
    
    test_points = [[1.5, 1.5], [4.5, 4.5], [2.5, 3.0]]
    test_preds = svm.predict(test_points)
    print(f"\n测试预测: {test_preds}")
    
    print("\n" + "="*60)
    print("核SVM测试 (RBF核)")
    print("="*60)
    
    kernel_svm = KernelSVM(kernel=Kernel.rbf, C=1.0, max_iterations=50)
    
    X_circle = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        [3.0, 3.0], [3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]
    ]
    y_circle = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    
    kernel_svm.fit(X_circle, y_circle)
    
    print(f"\n支持向量数量: {len(kernel_svm.support_vectors)}")
    
    circle_preds = kernel_svm.predict(X_circle)
    circle_accuracy = sum(1 for p, yt in zip(circle_preds, y_circle) if p == yt) / len(y_circle)
    print(f"训练准确率: {circle_accuracy:.2%}")
    
    print("\n" + "="*60)
    print("核函数测试")
    print("="*60)
    
    x1 = [1.0, 2.0]
    x2 = [3.0, 4.0]
    
    print(f"\n向量 x1={x1}, x2={x2}")
    print(f"线性核: {Kernel.linear(x1, x2):.2f}")
    print(f"多项式核(degree=2): {Kernel.polynomial(x1, x2, degree=2):.2f}")
    print(f"RBF核(gamma=0.1): {Kernel.rbf(x1, x2, gamma=0.1):.4f}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Cortes & Vapnik (1995) - SVM奠基论文")
    print("Vapnik (1995) - 统计学习理论基础")
    print("Burges (1998) - SVM教程综述")