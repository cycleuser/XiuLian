"""
连接主义方法实现（非深度学习）

参考文献:
- Rosenblatt (1958): 感知机
- Hopfield (1982): Hopfield网络
- Ackley, Hinton & Sejnowski (1985): Boltzmann机
"""

import numpy as np
from typing import List, Tuple, Optional
import random


class Perceptron:
    """感知机实现
    
    基于Rosenblatt (1958)的经典感知机算法
    
    参考:
    - Rosenblatt, F. (1958). "The perceptron: A probabilistic model for information storage and organization in the brain"
    - Minsky, M. & Papert, S. (1969). "Perceptrons: An introduction to computational geometry"
    - Novikoff, A.B.J. (1962). "On convergence proofs on perceptrons"
    
    特点:
    - 线性分类器
    - 在线学习
    - 收敛性保证（如果数据线性可分）
    
    时间复杂度:
    - 训练: O(n*m*k) 其中n为样本数，m为特征数，k为迭代次数
    - 推理: O(m)
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.1, max_iterations: int = 1000):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # 初始化权重
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # 统计信息
        self.iterations_used = 0
        self.converged = False
    
    def predict(self, x: np.ndarray) -> int:
        """预测 - O(m)
        
        线性组合 + 阈值函数
        """
        activation = np.dot(self.weights, x) + self.bias
        return 1 if activation >= 0 else -1
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """训练 - 感知机学习算法
        
        Novikoff定理: 如果数据线性可分，算法在有限步内收敛
        
        返回是否收敛
        """
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            errors = 0
            
            for i in range(n_samples):
                prediction = self.predict(X[i])
                
                # 更新规则: w = w + η * (y - y_hat) * x
                if prediction != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    errors += 1
            
            if errors == 0:
                self.converged = True
                self.iterations_used = iteration + 1
                return True
        
        self.iterations_used = self.max_iterations
        return False
    
    def decision_boundary(self, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """绘制决策边界（仅适用于2D）"""
        if self.n_features != 2:
            raise ValueError("仅支持2D可视化")
        
        x = np.linspace(x_range[0], x_range[1], 100)
        y = -(self.weights[0] * x + self.bias) / self.weights[1]
        return x, y


class HopfieldNetwork:
    """Hopfield网络实现
    
    基于Hopfield (1982)的联想记忆网络
    
    参考:
    - Hopfield, J.J. (1982). "Neural networks and physical systems with emergent collective computational abilities"
    - Hopfield, J.J. (1984). "Neurons with graded response have collective computational properties"
    - Tank, D.W. & Hopfield, J.J. (1986). "Simple 'neural' optimization networks"
    
    特点:
    - 联想记忆
    - 内容可寻址
    - 能量函数单调递减
    - 可用于优化问题
    
    时间复杂度:
    - 存储: O(n^2) 其中n为神经元数
    - 检索: O(n^2 * k) 其中k为迭代次数
    """
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.patterns_stored = 0
    
    def store_pattern(self, pattern: np.ndarray):
        """存储模式 - Hebb学习规则
        
        权重更新: w_ij = w_ij + x_i * x_j (i ≠ j)
        
        时间复杂度: O(n^2)
        """
        # 确保模式为+1/-1
        pattern = np.sign(pattern)
        pattern[pattern == 0] = 1
        
        # Hebb学习
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    self.weights[i, j] += pattern[i] * pattern[j]
        
        self.patterns_stored += 1
    
    def energy(self, state: np.ndarray) -> float:
        """计算能量函数
        
        E = -0.5 * Σ_i Σ_j w_ij * s_i * s_j
        
        能量单调递减，保证收敛
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def retrieve(self, probe: np.ndarray, max_iterations: int = 100, 
                 async_update: bool = True) -> Tuple[np.ndarray, int]:
        """检索模式 - 异步更新
        
        迭代更新直到收敛
        
        时间复杂度: O(n^2 * k)
        """
        state = probe.copy()
        
        for iteration in range(max_iterations):
            old_state = state.copy()
            
            if async_update:
                # 异步更新：逐个神经元更新
                for i in range(self.n_neurons):
                    activation = np.dot(self.weights[i], state)
                    state[i] = 1 if activation >= 0 else -1
            else:
                # 同步更新：所有神经元同时更新
                activations = np.dot(self.weights, state)
                state = np.sign(activations)
                state[state == 0] = 1
            
            # 检查收敛
            if np.array_equal(state, old_state):
                return state, iteration + 1
        
        return state, max_iterations
    
    def capacity_estimate(self) -> float:
        """估计存储容量
        
        理论上限: p_max ≈ 0.138 * n
        """
        return 0.138 * self.n_neurons


class BoltzmannMachine:
    """Boltzmann机实现
    
    基于Ackley, Hinton & Sejnowski (1985)
    
    参考:
    - Ackley, D.H., Hinton, G.E. & Sejnowski, T.J. (1985). "A learning algorithm for Boltzmann machines"
    - Hinton, G.E. & Sejnowski, T.J. (1983). "Optimal perceptual inference"
    - Smolensky, P. (1986). "Information processing in dynamical systems: Foundations of harmony theory"
    
    特点:
    - 随机神经网络
    - 概率生成模型
    - 隐藏单元学习内部表示
    - 模拟退火
    
    时间复杂度:
    - 训练: O(n^2 * k * s) 其中k为迭代次数，s为采样步数
    - 推理: O(n^2 * s)
    """
    
    def __init__(self, n_visible: int, n_hidden: int, temperature: float = 1.0):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden
        self.temperature = temperature
        
        # 初始化权重
        self.weights = np.random.randn(self.n_total, self.n_total) * 0.01
        np.fill_diagonal(self.weights, 0)  # 无自连接
        
        # 可见层和隐藏层的索引
        self.visible_indices = list(range(n_visible))
        self.hidden_indices = list(range(n_visible, self.n_total))
    
    def energy(self, state: np.ndarray) -> float:
        """计算能量
        
        E(v, h) = -Σ_i Σ_j w_ij * s_i * s_j
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def probability(self, i: int, state: np.ndarray) -> float:
        """计算神经元i激活概率
        
        P(s_i = 1) = 1 / (1 + exp(-Σ_j w_ij * s_j / T))
        """
        activation = np.dot(self.weights[i], state) / self.temperature
        return 1.0 / (1.0 + np.exp(-activation))
    
    def sample(self, n_steps: int = 1000) -> np.ndarray:
        """Gibbs采样
        
        从当前分布采样状态
        """
        state = np.random.choice([0, 1], size=self.n_total)
        
        for _ in range(n_steps):
            # 随机选择一个神经元更新
            i = random.randint(0, self.n_total - 1)
            prob = self.probability(i, state)
            state[i] = 1 if random.random() < prob else 0
        
        return state
    
    def contrastive_divergence(self, data: np.ndarray, learning_rate: float = 0.1, 
                                n_gibbs_steps: int = 1) -> None:
        """对比散度学习算法 (CD-k)
        
        近似最大似然学习
        
        时间复杂度: O(n^2 * k)
        """
        n_samples = data.shape[0]
        
        # 正相：计算数据统计
        positive_stats = np.zeros((self.n_total, self.n_total))
        for sample in data:
            # 设置可见层
            state = np.zeros(self.n_total)
            state[:self.n_visible] = sample
            
            # 采样隐藏层
            for i in self.hidden_indices:
                prob = self.probability(i, state)
                state[i] = 1 if random.random() < prob else 0
            
            # 计算相关性
            for i in range(self.n_total):
                for j in range(i+1, self.n_total):
                    positive_stats[i, j] += state[i] * state[j]
                    positive_stats[j, i] = positive_stats[i, j]
        
        positive_stats /= n_samples
        
        # 负相：采样重构
        negative_stats = np.zeros((self.n_total, self.n_total))
        for sample in data:
            state = np.zeros(self.n_total)
            state[:self.n_visible] = sample
            
            # Gibbs采样
            for _ in range(n_gibbs_steps):
                # 更新隐藏层
                for i in self.hidden_indices:
                    prob = self.probability(i, state)
                    state[i] = 1 if random.random() < prob else 0
                
                # 更新可见层
                for i in self.visible_indices:
                    prob = self.probability(i, state)
                    state[i] = 1 if random.random() < prob else 0
            
            # 计算相关性
            for i in range(self.n_total):
                for j in range(i+1, self.n_total):
                    negative_stats[i, j] += state[i] * state[j]
                    negative_stats[j, i] = negative_stats[i, j]
        
        negative_stats /= n_samples
        
        # 更新权重
        self.weights += learning_rate * (positive_stats - negative_stats)
        
        # 保持对称和无自连接
        self.weights = (self.weights + self.weights.T) / 2
        np.fill_diagonal(self.weights, 0)
    
    def reconstruct(self, visible: np.ndarray, n_steps: int = 100) -> np.ndarray:
        """重构可见层
        
        用于测试生成能力
        """
        state = np.zeros(self.n_total)
        state[:self.n_visible] = visible
        
        # 固定可见层，采样隐藏层
        for _ in range(n_steps):
            for i in self.hidden_indices:
                prob = self.probability(i, state)
                state[i] = 1 if random.random() < prob else 0
        
        # 重构可见层
        for i in self.visible_indices:
            prob = self.probability(i, state)
            state[i] = 1 if random.random() < prob else 0
        
        return state[:self.n_visible]


# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("连接主义方法测试（非深度学习）")
    print("="*60)
    
    # 1. 感知机测试
    print("\n1. 感知机 (Perceptron)")
    print("-"*40)
    
    # 创建线性可分数据
    X = np.array([
        [1, 1], [1.5, 1.5], [2, 2],  # 正类
        [-1, -1], [-1.5, -1.5], [-2, -2]  # 负类
    ])
    y = np.array([1, 1, 1, -1, -1, -1])
    
    perceptron = Perceptron(n_features=2)
    converged = perceptron.train(X, y)
    
    print(f"收敛: {converged}")
    print(f"迭代次数: {perceptron.iterations_used}")
    print(f"权重: {perceptron.weights}")
    print(f"偏置: {perceptron.bias}")
    
    # 测试
    test_point = np.array([1.2, 1.3])
    prediction = perceptron.predict(test_point)
    print(f"测试点 {test_point} 预测: {'正类' if prediction == 1 else '负类'}")
    
    # 2. Hopfield网络测试
    print("\n2. Hopfield网络 (Hopfield Network)")
    print("-"*40)
    
    hopfield = HopfieldNetwork(n_neurons=10)
    
    # 存储模式
    pattern1 = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1])
    pattern2 = np.array([-1, -1, 1, 1, 1, 1, -1, -1, -1, -1])
    
    hopfield.store_pattern(pattern1)
    hopfield.store_pattern(pattern2)
    
    print(f"存储模式数: {hopfield.patterns_stored}")
    print(f"理论容量: {hopfield.capacity_estimate():.2f}")
    
    # 检索测试
    noisy_pattern = pattern1.copy()
    noisy_pattern[0] = -1  # 加入噪声
    noisy_pattern[1] = -1
    
    retrieved, iterations = hopfield.retrieve(noisy_pattern)
    
    print(f"原始模式: {pattern1[:5]}...")
    print(f"噪声输入: {noisy_pattern[:5]}...")
    print(f"检索结果: {retrieved[:5]}...")
    print(f"迭代次数: {iterations}")
    print(f"检索成功: {np.array_equal(retrieved, pattern1)}")
    
    # 3. Boltzmann机测试
    print("\n3. Boltzmann机 (Boltzmann Machine)")
    print("-"*40)
    
    bm = BoltzmannMachine(n_visible=4, n_hidden=2, temperature=1.0)
    
    # 训练数据（简单模式）
    data = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    
    # 训练
    for epoch in range(10):
        bm.contrastive_divergence(data, learning_rate=0.1, n_gibbs_steps=10)
    
    # 重构测试
    test_input = np.array([1, 1, 0, 0])
    reconstructed = bm.reconstruct(test_input)
    
    print(f"输入: {test_input}")
    print(f"重构: {reconstructed.astype(int)}")