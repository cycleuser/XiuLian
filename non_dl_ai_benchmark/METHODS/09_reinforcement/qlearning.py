#!/usr/bin/env python3
"""
强化学习实现

参考文献:
@phdthesis{watkins1989,
  title={Learning from delayed rewards},
  author={Watkins, Christopher J. C. H.},
  year={1989},
  school={King's College, Cambridge}
}

@article{sutton1988,
  title={Learning to predict by the methods of temporal differences},
  author={Sutton, Richard S.},
  journal={Machine Learning},
  volume={3},
  number={1},
  pages={9--44},
  year={1988}
}

@book{sutton2018,
  title={Reinforcement Learning: An Introduction},
  author={Sutton, Richard S. and Barto, Andrew G.},
  publisher={MIT Press},
  year={2018}
}
"""

from typing import Dict, List, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math


@dataclass
class State:
    name: str
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class Action:
    name: str
    q_value: float = 0.0


@dataclass
class Transition:
    next_state: str
    reward: float
    probability: float


class Environment:
    """
    强化学习环境
    
    定义状态空间、动作空间和奖励函数
    """
    
    def __init__(self):
        self.states: Set[str] = set()
        self.actions: Dict[str, Set[str]] = defaultdict(set)
        self.transitions: Dict[str, Dict[str, List[Transition]]] = defaultdict(lambda: defaultdict(list))
        self.current_state: str = ""
        self.terminal_states: Set[str] = set()
    
    def add_state(self, name: str, terminal: bool = False):
        self.states.add(name)
        if terminal:
            self.terminal_states.add(name)
    
    def add_action(self, state: str, action: str):
        self.actions[state].add(action)
    
    def add_transition(self, state: str, action: str, next_state: str, 
                       reward: float, probability: float = 1.0):
        self.transitions[state][action].append(
            Transition(next_state=next_state, reward=reward, probability=probability)
        )
    
    def reset(self, initial_state: str) -> str:
        self.current_state = initial_state
        return self.current_state
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        if self.current_state not in self.transitions or \
           action not in self.transitions[self.current_state]:
            return self.current_state, 0.0, False
        
        transitions = self.transitions[self.current_state][action]
        
        total_prob = sum(t.probability for t in transitions)
        r = random.random() * total_prob
        
        cumulative = 0.0
        for t in transitions:
            cumulative += t.probability
            if r <= cumulative:
                self.current_state = t.next_state
                done = t.next_state in self.terminal_states
                return t.next_state, t.reward, done
        
        return self.current_state, 0.0, False
    
    def get_actions(self, state: str) -> List[str]:
        return list(self.actions[state])


class QAgent:
    """
    Q-Learning智能体
    
    基于 Watkins (1989) 的Q-Learning算法
    使用时序差分学习更新Q值
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def get_q(self, state: str, action: str) -> float:
        return self.q_table[state][action]
    
    def get_best_action(self, state: str, actions: List[str]) -> str:
        if not actions:
            return ""
        
        q_values = {a: self.get_q(state, a) for a in actions}
        return max(q_values, key=q_values.get)
    
    def select_action(self, state: str, actions: List[str]) -> str:
        if not actions:
            return ""
        
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        return self.get_best_action(state, actions)
    
    def update(self, state: str, action: str, reward: float, 
               next_state: str, next_actions: List[str], done: bool):
        current_q = self.get_q(state, action)
        
        if done:
            target = reward
        else:
            max_next_q = max(self.get_q(next_state, a) for a in next_actions) if next_actions else 0
            target = reward + self.gamma * max_next_q
        
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action] = new_q
        
        self.epsilon *= self.epsilon_decay
    
    def train(self, env: Environment, episodes: int = 1000, 
              max_steps: int = 100, initial_state: str = "") -> List[float]:
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset(initial_state)
            total_reward = 0.0
            
            for step in range(max_steps):
                actions = env.get_actions(state)
                action = self.select_action(state, actions)
                
                next_state, reward, done = env.step(action)
                next_actions = env.get_actions(next_state)
                
                self.update(state, action, reward, next_state, next_actions, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards_history.append(total_reward)
        
        return rewards_history


class SarsaAgent:
    """
    SARSA智能体
    
    基于 Sutton (1988) 的时序差分学习
    与Q-Learning不同，SARSA使用实际采取的动作更新Q值
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def get_q(self, state: str, action: str) -> float:
        return self.q_table[state][action]
    
    def select_action(self, state: str, actions: List[str]) -> str:
        if not actions:
            return ""
        
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        q_values = {a: self.get_q(state, a) for a in actions}
        return max(q_values, key=q_values.get)
    
    def update(self, state: str, action: str, reward: float,
               next_state: str, next_action: str, done: bool):
        current_q = self.get_q(state, action)
        
        if done:
            target = reward
        else:
            next_q = self.get_q(next_state, next_action)
            target = reward + self.gamma * next_q
        
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action] = new_q
        
        self.epsilon *= self.epsilon_decay
    
    def train(self, env: Environment, episodes: int = 1000,
              max_steps: int = 100, initial_state: str = "") -> List[float]:
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset(initial_state)
            actions = env.get_actions(state)
            action = self.select_action(state, actions)
            total_reward = 0.0
            
            for step in range(max_steps):
                next_state, reward, done = env.step(action)
                next_actions = env.get_actions(next_state)
                next_action = self.select_action(next_state, next_actions)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                total_reward += reward
                state = next_state
                action = next_action
                
                if done:
                    break
            
            rewards_history.append(total_reward)
        
        return rewards_history


class PolicyGradientAgent:
    """
    策略梯度智能体
    
    基于 Sutton & Barto (2018) 的REINFORCE算法
    直接优化策略参数
    """
    
    def __init__(self, alpha: float = 0.01, gamma: float = 0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.policy: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def get_action_prob(self, state: str, action: str, actions: List[str]) -> float:
        logits = {a: self.policy[state][a] for a in actions}
        max_logit = max(logits.values())
        exp_logits = {a: math.exp(l - max_logit) for a, l in logits.items()}
        total = sum(exp_logits.values())
        return exp_logits[action] / total
    
    def select_action(self, state: str, actions: List[str]) -> str:
        if not actions:
            return ""
        
        probs = [self.get_action_prob(state, a, actions) for a in actions]
        return random.choices(actions, weights=probs)[0]
    
    def update(self, trajectory: List[Tuple[str, str, float]]):
        G = 0.0
        for state, action, reward in reversed(trajectory):
            G = self.gamma * G + reward
            
            actions = list(self.policy[state].keys())
            if not actions:
                continue
            
            prob = self.get_action_prob(state, action, actions)
            
            gradient = 1.0 / prob
            self.policy[state][action] += self.alpha * G * gradient
    
    def train(self, env: Environment, episodes: int = 1000,
              max_steps: int = 100, initial_state: str = "") -> List[float]:
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset(initial_state)
            trajectory = []
            total_reward = 0.0
            
            for step in range(max_steps):
                actions = env.get_actions(state)
                action = self.select_action(state, actions)
                
                next_state, reward, done = env.step(action)
                trajectory.append((state, action, reward))
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.update(trajectory)
            rewards_history.append(total_reward)
        
        return rewards_history


class ValueIteration:
    """
    值迭代算法
    
    基于 Sutton & Barto (2018) 的动态规划方法
    用于已知环境模型的规划
    """
    
    def __init__(self, gamma: float = 0.9, theta: float = 1e-6):
        self.gamma = gamma
        self.theta = theta
        self.values: Dict[str, float] = {}
        self.policy: Dict[str, str] = {}
    
    def solve(self, env: Environment):
        for state in env.states:
            self.values[state] = 0.0
        
        iteration = 0
        while True:
            delta = 0.0
            
            for state in env.states:
                if state in env.terminal_states:
                    continue
                
                v = self.values[state]
                actions = env.get_actions(state)
                
                if not actions:
                    continue
                
                action_values = []
                for action in actions:
                    transitions = env.transitions[state][action]
                    q_value = sum(
                        t.probability * (t.reward + self.gamma * self.values[t.next_state])
                        for t in transitions
                    )
                    action_values.append((action, q_value))
                
                best_action, best_value = max(action_values, key=lambda x: x[1])
                self.values[state] = best_value
                self.policy[state] = best_action
                
                delta = max(delta, abs(v - self.values[state]))
            
            iteration += 1
            if delta < self.theta:
                break
        
        return iteration


def create_grid_world() -> Environment:
    """
    创建网格世界环境
    
    经典的强化学习测试环境
    """
    env = Environment()
    
    for i in range(4):
        for j in range(4):
            state = f"({i},{j})"
            env.add_state(state)
            
            if i > 0:
                env.add_action(state, "up")
                env.add_transition(state, "up", f"({i-1},{j})", -1.0)
            if i < 3:
                env.add_action(state, "down")
                env.add_transition(state, "down", f"({i+1},{j})", -1.0)
            if j > 0:
                env.add_action(state, "left")
                env.add_transition(state, "left", f"({i},{j-1})", -1.0)
            if j < 3:
                env.add_action(state, "right")
                env.add_transition(state, "right", f"({i},{j+1})", -1.0)
    
    env.add_state("(0,3)", terminal=True)
    env.add_state("(3,3)", terminal=True)
    env.add_transition("(0,2)", "up", "(0,3)", 10.0)
    env.add_transition("(3,2)", "up", "(3,3)", 10.0)
    
    return env


if __name__ == "__main__":
    print("="*60)
    print("Q-Learning测试 - 网格世界")
    print("="*60)
    
    env = create_grid_world()
    q_agent = QAgent(alpha=0.1, gamma=0.9, epsilon=0.3, epsilon_decay=0.99)
    
    rewards = q_agent.train(env, episodes=500, max_steps=50, initial_state="(0,0)")
    
    print(f"\n训练500回合")
    print(f"最后10回合平均奖励: {sum(rewards[-10:]) / 10:.2f}")
    
    print("\n学到的策略:")
    for state in sorted(env.states):
        if state not in env.terminal_states:
            actions = env.get_actions(state)
            if actions:
                best = q_agent.get_best_action(state, actions)
                print(f"  状态 {state}: 最佳动作 = {best}")
    
    print("\n" + "="*60)
    print("值迭代测试")
    print("="*60)
    
    vi = ValueIteration(gamma=0.9)
    iterations = vi.solve(env)
    
    print(f"\n收敛迭代次数: {iterations}")
    print("\n状态值:")
    for state in sorted(env.states):
        print(f"  {state}: {vi.values[state]:.2f}")
    
    print("\n最优策略:")
    for state in sorted(env.states):
        if state in vi.policy:
            print(f"  {state}: {vi.policy[state]}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Watkins (1989) - Q-Learning算法起源")
    print("Sutton (1988) - 时序差分学习基础")
    print("Sutton & Barto (2018) - 强化学习经典教材")