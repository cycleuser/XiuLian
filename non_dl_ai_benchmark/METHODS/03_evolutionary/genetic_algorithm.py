"""
进化计算方法实现

参考文献:
- Holland (1975): 遗传算法
- Koza (1992): 遗传编程
- Rechenberg (1973): 演化策略
"""

import numpy as np
from typing import List, Tuple, Callable, Any
import random
from copy import deepcopy


class GeneticAlgorithm:
    """遗传算法实现
    
    基于Holland (1975)的经典遗传算法
    
    参考:
    - Holland, J.H. (1975). "Adaptation in Natural and Artificial Systems"
    - Goldberg, D.E. & Holland, J.H. (1988). "Genetic algorithms and machine learning"
    - Mitchell, M. (1998). "An Introduction to Genetic Algorithms"
    
    核心操作:
    1. 选择 - 适应度比例选择
    2. 交叉 - 单点/多点交叉
    3. 变异 - 位翻转变异
    
    时间复杂度:
    - 每代: O(n * m) 其中n为种群大小，m为染色体长度
    """
    
    def __init__(self, 
                 fitness_func: Callable,
                 gene_length: int,
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7,
                 elite_size: int = 2):
        self.fitness_func = fitness_func
        self.gene_length = gene_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
    
    def initialize_population(self):
        """初始化种群"""
        self.population = [
            [random.randint(0, 1) for _ in range(self.gene_length)]
            for _ in range(self.population_size)
        ]
    
    def evaluate_fitness(self) -> List[float]:
        """评估适应度"""
        return [self.fitness_func(individual) for individual in self.population]
    
    def selection(self, fitnesses: List[float]) -> List[List[int]]:
        """轮盘赌选择
        
        适应度越高的个体被选中概率越大
        """
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return [random.choice(self.population) for _ in range(self.population_size)]
        
        probabilities = [f / total_fitness for f in fitnesses]
        
        selected = []
        for _ in range(self.population_size - self.elite_size):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(self.population[i].copy())
                    break
        
        return selected
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """单点交叉"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, self.gene_length - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual: List[int]) -> List[int]:
        """位翻转变异"""
        mutated = individual.copy()
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def evolve(self, max_generations: int = 100, target_fitness: float = None) -> Tuple[List[int], float]:
        """进化主循环"""
        if not self.population:
            self.initialize_population()
        
        for generation in range(max_generations):
            # 评估适应度
            fitnesses = self.evaluate_fitness()
            
            # 记录历史
            max_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
            self.fitness_history.append((max_fit, avg_fit))
            
            # 更新最优个体
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            # 检查目标
            if target_fitness and self.best_fitness >= target_fitness:
                return self.best_individual, self.best_fitness
            
            # 精英保留
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [self.population[i].copy() for i in elite_indices]
            
            # 选择
            selected = self.selection(fitnesses)
            
            # 交叉和变异
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))
            
            self.population = new_population
        
        return self.best_individual, self.best_fitness


class GeneticProgramming:
    """遗传编程实现
    
    基于Koza (1992)的遗传编程
    
    参考:
    - Koza, J.R. (1992). "Genetic Programming: On the Programming of Computers by Means of Natural Selection"
    - Banzhaf, W. et al. (1998). "Genetic Programming: An Introduction"
    
    特点:
    - 程序树表示
    - 函数集 + 终端集
    - 树交叉和变异
    """
    
    def __init__(self, 
                 function_set: List[Callable],
                 terminal_set: List[Any],
                 max_depth: int = 5,
                 population_size: int = 100):
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.max_depth = max_depth
        self.population_size = population_size
        
        self.population = []
        self.best_program = None
    
    class TreeNode:
        """程序树节点"""
        def __init__(self, value, children=None):
            self.value = value
            self.children = children or []
        
        def evaluate(self, **kwargs):
            """执行程序"""
            if callable(self.value):
                args = [child.evaluate(**kwargs) for child in self.children]
                return self.value(*args)
            return self.value
        
        def copy(self):
            """深拷贝"""
            return self.__class__(
                self.value,
                [child.copy() for child in self.children]
            )
        
        def depth(self) -> int:
            """树深度"""
            if not self.children:
                return 1
            return 1 + max(child.depth() for child in self.children)
    
    def generate_random_tree(self, max_depth: int = None, method: str = 'grow') -> TreeNode:
        """生成随机程序树
        
        method:
        - 'grow': 变长树
        - 'full': 满树
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        if max_depth == 1 or (method == 'grow' and random.random() < 0.3):
            # 终端节点
            return self.TreeNode(random.choice(self.terminal_set))
        else:
            # 函数节点
            func = random.choice(self.function_set)
            n_children = func.__code__.co_argcount if hasattr(func, '__code__') else 2
            children = [
                self.generate_random_tree(max_depth - 1, method)
                for _ in range(n_children)
            ]
            return self.TreeNode(func, children)
    
    def crossover(self, tree1: TreeNode, tree2: TreeNode) -> TreeNode:
        """树交叉"""
        child = tree1.copy()
        
        # 随机选择一个子树替换
        if random.random() < 0.9:  # 90%概率交叉
            # 简化实现：随机选择深度
            subtree = tree2.copy()
            # 实际实现需要更复杂的子树选择
            
        return child
    
    def mutate(self, tree: TreeNode, mutation_rate: float = 0.1) -> TreeNode:
        """树变异"""
        mutated = tree.copy()
        
        if random.random() < mutation_rate:
            # 生成新子树替换
            new_subtree = self.generate_random_tree(max_depth=3)
            # 实际实现需要随机选择节点替换
        
        return mutated


class EvolutionStrategy:
    """演化策略实现
    
    基于Rechenberg (1973)和Schwefel (1977)
    
    参考:
    - Rechenberg, I. (1973). "Evolutionsstrategie"
    - Schwefel, H.P. (1977). "Numerische Optimierung von Computer-Modellen"
    - Beyer, H.G. & Schwefel, H.P. (2002). "Evolution strategies: A comprehensive introduction"
    
    特点:
    - 实数编码
    - 自适应步长
    - (μ, λ) 或 (μ + λ) 选择
    """
    
    def __init__(self, 
                 objective_func: Callable,
                 n_dimensions: int,
                 mu: int = 15,
                 lambda_: int = 100,
                 plus_selection: bool = True):
        self.objective_func = objective_func
        self.n_dimensions = n_dimensions
        self.mu = mu
        self.lambda_ = lambda_
        self.plus_selection = plus_selection
        
        self.population = []
        self.strategies = []  # 步长
    
    def initialize(self, bounds: List[Tuple[float, float]]):
        """初始化种群和步长"""
        self.population = [
            np.array([
                random.uniform(bounds[d][0], bounds[d][1])
                for d in range(self.n_dimensions)
            ])
            for _ in range(self.mu)
        ]
        
        self.strategies = [
            np.ones(self.n_dimensions) * 0.5  # 初始步长
            for _ in range(self.mu)
        ]
    
    def mutate(self, parent: np.ndarray, strategy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """变异操作
        
        步长自适应: σ' = σ * exp(τ * N(0,1))
        """
        # 全局学习率
        tau = 1.0 / np.sqrt(2 * self.n_dimensions)
        
        # 变异步长
        new_strategy = strategy * np.exp(tau * np.random.randn())
        
        # 变异个体
        offspring = parent + new_strategy * np.random.randn(self.n_dimensions)
        
        return offspring, new_strategy
    
    def evolve(self, max_generations: int = 100) -> Tuple[np.ndarray, float]:
        """演化主循环"""
        for gen in range(max_generations):
            # 生成子代
            offspring = []
            offspring_strategies = []
            
            for _ in range(self.lambda_):
                # 随机选择父代
                idx = random.randint(0, self.mu - 1)
                child, child_strategy = self.mutate(
                    self.population[idx], 
                    self.strategies[idx]
                )
                offspring.append(child)
                offspring_strategies.append(child_strategy)
            
            # 评估
            fitnesses = [self.objective_func(ind) for ind in offspring]
            
            # 选择
            if self.plus_selection:
                # (μ + λ) 选择
                combined = list(zip(
                    self.population + offspring,
                    self.strategies + offspring_strategies,
                    [self.objective_func(ind) for ind in self.population] + fitnesses
                ))
                combined.sort(key=lambda x: -x[2])
                
                self.population = [c[0] for c in combined[:self.mu]]
                self.strategies = [c[1] for c in combined[:self.mu]]
            else:
                # (μ, λ) 选择
                combined = list(zip(offspring, offspring_strategies, fitnesses))
                combined.sort(key=lambda x: -x[2])
                
                self.population = [c[0] for c in combined[:self.mu]]
                self.strategies = [c[1] for c in combined[:self.mu]]
        
        # 返回最优
        fitnesses = [self.objective_func(ind) for ind in self.population]
        best_idx = np.argmax(fitnesses)
        return self.population[best_idx], fitnesses[best_idx]


# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("进化计算方法测试")
    print("="*60)
    
    # 1. 遗传算法测试 - OneMax问题
    print("\n1. 遗传算法 (Genetic Algorithm)")
    print("-"*40)
    
    def onemax(individual):
        return sum(individual)
    
    ga = GeneticAlgorithm(
        fitness_func=onemax,
        gene_length=20,
        population_size=50,
        mutation_rate=0.01,
        crossover_rate=0.7
    )
    
    best, fitness = ga.evolve(max_generations=100, target_fitness=20)
    
    print(f"最优个体: {best}")
    print(f"最优适应度: {fitness}")
    print(f"迭代次数: {len(ga.fitness_history)}")
    
    # 2. 演化策略测试 - 球函数优化
    print("\n2. 演化策略 (Evolution Strategy)")
    print("-"*40)
    
    def sphere(x):
        return -sum(x ** 2)  # 最小化 -> 最大化
    
    es = EvolutionStrategy(
        objective_func=sphere,
        n_dimensions=5,
        mu=15,
        lambda_=100
    )
    
    bounds = [(-5, 5)] * 5
    es.initialize(bounds)
    
    best, fitness = es.evolve(max_generations=50)
    
    print(f"最优解: {best}")
    print(f"最优值: {-fitness}")  # 转回最小值