#!/usr/bin/env python3
"""
图推理实现

参考文献:
@article{quillian1966,
  title={Semantic memory},
  author={Quillian, M. Ross},
  booktitle={Semantic Information Processing},
  pages={227--270},
  year={1968}
}

@article{brachman1985,
  title={On the epistemological status of semantic networks},
  author={Brachman, Ronald J.},
  booktitle={Readings in Knowledge Representation},
  pages={191--215},
  year={1985}
}

@article{singhal2012,
  title={Introducing the Knowledge Graph: things, not strings},
  author={Singhal, Amit},
  journal={Google Official Blog},
  year={2012}
}

@article{bordes2013,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2787--2795},
  year={2013}
}
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    name: str
    type: str = "entity"
    properties: Dict[str, Any] = field(default_factory=dict)


class SemanticNetwork:
    """
    语义网络
    
    基于 Quillian (1968) 的语义记忆模型
    使用图结构表示概念之间的关系
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.outgoing: Dict[str, List[Edge]] = defaultdict(list)
        self.incoming: Dict[str, List[Edge]] = defaultdict(list)
    
    def add_node(self, name: str, type: str = "entity", properties: Dict[str, Any] = None):
        self.nodes[name] = Node(name=name, type=type, properties=properties or {})
    
    def add_edge(self, source: str, relation: str, target: str, 
                 weight: float = 1.0, properties: Dict[str, Any] = None):
        edge = Edge(source=source, relation=relation, target=target, 
                    weight=weight, properties=properties or {})
        self.edges.append(edge)
        self.outgoing[source].append(edge)
        self.incoming[target].append(edge)
    
    def get_node(self, name: str) -> Optional[Node]:
        return self.nodes.get(name)
    
    def get_relations(self, node: str, direction: str = "out") -> List[Edge]:
        if direction == "out":
            return self.outgoing[node]
        return self.incoming[node]
    
    def query(self, source: str, relation: str = None, target: str = None) -> List[Edge]:
        results = []
        for edge in self.outgoing[source]:
            if relation and edge.relation != relation:
                continue
            if target and edge.target != target:
                continue
            results.append(edge)
        return results
    
    def spread_activation(self, start_nodes: List[str], decay: float = 0.8, 
                          threshold: float = 0.1, max_depth: int = 5) -> Dict[str, float]:
        """
        激活扩散算法
        
        基于 Quillian (1968) 的概念激活扩散
        从起始节点激活相关概念
        """
        activation: Dict[str, float] = {n: 1.0 for n in start_nodes}
        visited: Set[str] = set(start_nodes)
        
        current_nodes = start_nodes
        
        for depth in range(max_depth):
            next_activation: Dict[str, float] = {}
            
            for node in current_nodes:
                if node not in activation:
                    continue
                
                node_activation = activation[node] * decay
                
                for edge in self.outgoing[node]:
                    target = edge.target
                    propagated = node_activation * edge.weight
                    
                    if target not in next_activation:
                        next_activation[target] = 0.0
                    next_activation[target] = max(next_activation[target], propagated)
            
            new_nodes = [n for n in next_activation if next_activation[n] >= threshold and n not in visited]
            
            for node, act in next_activation.items():
                if act >= threshold:
                    activation[node] = act
                    visited.add(node)
            
            current_nodes = new_nodes
            
            if not new_nodes:
                break
        
        return activation
    
    def find_path(self, start: str, end: str, max_depth: int = 10) -> List[Edge]:
        """
        路径查找
        
        使用BFS寻找两个节点之间的路径
        """
        if start not in self.nodes or end not in self.nodes:
            return []
        
        queue: List[Tuple[str, List[Edge]]] = [(start, [])]
        visited: Set[str] = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end:
                return path
            
            if len(path) >= max_depth:
                continue
            
            for edge in self.outgoing[current]:
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge]))
        
        return []
    
    def get_neighbors(self, node: str, distance: int = 1) -> Set[str]:
        """
        获取邻域节点
        """
        neighbors: Set[str] = {node}
        
        for _ in range(distance):
            new_neighbors: Set[str] = set()
            for n in neighbors:
                for edge in self.outgoing[n]:
                    new_neighbors.add(edge.target)
                for edge in self.incoming[n]:
                    new_neighbors.add(edge.source)
            neighbors.update(new_neighbors)
        
        neighbors.discard(node)
        return neighbors


class KnowledgeGraph(SemanticNetwork):
    """
    知识图谱
    
    基于 Singhal (2012) 的知识图谱概念
    扩展语义网络，支持更复杂的推理
    """
    
    def __init__(self):
        super().__init__()
        self.entity_types: Dict[str, Set[str]] = defaultdict(set)
        self.relation_hierarchy: Dict[str, str] = {}
    
    def add_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None):
        self.add_node(name, type=entity_type, properties=properties)
        self.entity_types[entity_type].add(name)
    
    def add_relation(self, source: str, relation: str, target: str,
                     weight: float = 1.0, properties: Dict[str, Any] = None):
        self.add_edge(source, relation, target, weight, properties)
    
    def get_entities_by_type(self, entity_type: str) -> List[Node]:
        names = self.entity_types.get(entity_type, set())
        return [self.nodes[n] for n in names if n in self.nodes]
    
    def transitive_query(self, source: str, relation: str, max_depth: int = 5) -> List[str]:
        """
        传递性查询
        
        查找所有通过传递关系可达的目标
        例如: A -> B -> C, 查询A的"part_of"返回[B, C]
        """
        results: Set[str] = set()
        visited: Set[str] = {source}
        queue = [source]
        
        for _ in range(max_depth):
            if not queue:
                break
            
            current = queue.pop(0)
            
            for edge in self.outgoing[current]:
                if edge.relation == relation and edge.target not in visited:
                    results.add(edge.target)
                    visited.add(edge.target)
                    queue.append(edge.target)
        
        return list(results)
    
    def infer_property(self, entity: str, property_relation: str) -> Any:
        """
        属性继承推理
        
        从祖先节点继承属性
        """
        if entity not in self.nodes:
            return None
        
        entity_node = self.nodes[entity]
        if property_relation in entity_node.properties:
            return entity_node.properties[property_relation]
        
        for edge in self.outgoing[entity]:
            if edge.relation == "is_a" or edge.relation == "subclass_of":
                inherited = self.infer_property(edge.target, property_relation)
                if inherited is not None:
                    return inherited
        
        return None
    
    def check_relation(self, source: str, target: str, relation: str) -> bool:
        """
        关系检查
        
        检查两个实体之间是否存在特定关系
        """
        for edge in self.outgoing[source]:
            if edge.target == target and edge.relation == relation:
                return True
        return False
    
    def get_common_ancestors(self, entity1: str, entity2: str) -> List[str]:
        """
        获取共同祖先
        
        查找两个实体的共同父类
        """
        ancestors1 = self._get_all_ancestors(entity1)
        ancestors2 = self._get_all_ancestors(entity2)
        return list(ancestors1 & ancestors2)
    
    def _get_all_ancestors(self, entity: str) -> Set[str]:
        ancestors: Set[str] = set()
        queue = [entity]
        
        while queue:
            current = queue.pop(0)
            for edge in self.outgoing[current]:
                if edge.relation in ("is_a", "subclass_of", "instance_of"):
                    if edge.target not in ancestors:
                        ancestors.add(edge.target)
                        queue.append(edge.target)
        
        return ancestors


class GraphEmbedding:
    """
    图嵌入
    
    基于 Bordes et al. (2013) 的TransE算法
    将实体和关系嵌入到低维向量空间
    """
    
    def __init__(self, embedding_dim: int = 50, margin: float = 1.0):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.entity_embeddings: Dict[str, List[float]] = {}
        self.relation_embeddings: Dict[str, List[float]] = {}
    
    def initialize_embeddings(self, entities: List[str], relations: List[str]):
        import random
        
        for entity in entities:
            self.entity_embeddings[entity] = [
                random.uniform(-0.1, 0.1) for _ in range(self.embedding_dim)
            ]
        
        for relation in relations:
            self.relation_embeddings[relation] = [
                random.uniform(-0.1, 0.1) for _ in range(self.embedding_dim)
            ]
    
    def normalize_embeddings(self):
        for entity, emb in self.entity_embeddings.items():
            norm = math.sqrt(sum(x ** 2 for x in emb))
            if norm > 0:
                self.entity_embeddings[entity] = [x / norm for x in emb]
    
    def score(self, source: str, relation: str, target: str) -> float:
        """
        TransE评分函数
        
        d(h + r, t) 应该小对于正样本，大对于负样本
        """
        if source not in self.entity_embeddings or \
           relation not in self.relation_embeddings or \
           target not in self.entity_embeddings:
            return float('inf')
        
        h = self.entity_embeddings[source]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[target]
        
        predicted = [h[i] + r[i] for i in range(self.embedding_dim)]
        
        distance = sum((predicted[i] - t[i]) ** 2 for i in range(self.embedding_dim))
        
        return math.sqrt(distance)
    
    def train_transe(self, positive_triples: List[Tuple[str, str, str]], 
                     entities: List[str], epochs: int = 100, 
                     learning_rate: float = 0.01):
        """
        TransE训练
        
        使用SGD优化嵌入
        """
        import random
        
        relations = list(set(r for _, r, _ in positive_triples))
        self.initialize_embeddings(entities, relations)
        
        for epoch in range(epochs):
            for h, r, t in positive_triples:
                h_emb = self.entity_embeddings[h]
                r_emb = self.relation_embeddings[r]
                t_emb = self.entity_embeddings[t]
                
                negative_entity = random.choice(entities)
                while negative_entity == h or negative_entity == t:
                    negative_entity = random.choice(entities)
                
                pos_score = self.score(h, r, t)
                neg_score = self.score(h, r, negative_entity)
                
                if pos_score + self.margin > neg_score:
                    gradient = []
                    for i in range(self.embedding_dim):
                        diff = h_emb[i] + r_emb[i] - t_emb[i]
                        grad = 2 * diff
                        gradient.append(grad)
                    
                    for i in range(self.embedding_dim):
                        h_emb[i] -= learning_rate * gradient[i]
                        r_emb[i] -= learning_rate * gradient[i]
                        t_emb[i] += learning_rate * gradient[i]
            
            self.normalize_embeddings()
    
    def predict_links(self, source: str, relation: str, 
                      candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        链接预测
        
        预测给定源实体和关系的目标实体
        """
        scores = []
        for candidate in candidates:
            score = self.score(source, relation, candidate)
            scores.append((candidate, score))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]


def create_common_knowledge_graph() -> KnowledgeGraph:
    """
    创建通用知识图谱示例
    """
    kg = KnowledgeGraph()
    
    kg.add_entity("Animal", "Class")
    kg.add_entity("Mammal", "Class")
    kg.add_entity("Bird", "Class")
    kg.add_entity("Dog", "Class")
    kg.add_entity("Cat", "Class")
    kg.add_entity("Eagle", "Class")
    
    kg.add_entity("Fido", "Individual")
    kg.add_entity("Whiskers", "Individual")
    
    kg.add_relation("Mammal", "is_a", "Animal")
    kg.add_relation("Bird", "is_a", "Animal")
    kg.add_relation("Dog", "is_a", "Mammal")
    kg.add_relation("Cat", "is_a", "Mammal")
    kg.add_relation("Eagle", "is_a", "Bird")
    
    kg.add_relation("Fido", "instance_of", "Dog")
    kg.add_relation("Whiskers", "instance_of", "Cat")
    
    kg.nodes["Dog"].properties["has_fur"] = True
    kg.nodes["Cat"].properties["has_fur"] = True
    kg.nodes["Mammal"].properties["warm_blooded"] = True
    kg.nodes["Bird"].properties["has_feathers"] = True
    
    return kg


if __name__ == "__main__":
    print("="*60)
    print("语义网络测试")
    print("="*60)
    
    sn = SemanticNetwork()
    
    sn.add_node("bird")
    sn.add_node("canary")
    sn.add_node("animal")
    
    sn.add_edge("canary", "is_a", "bird")
    sn.add_edge("bird", "is_a", "animal")
    sn.add_edge("canary", "has_color", "yellow", weight=0.9)
    sn.add_edge("canary", "can", "sing", weight=0.8)
    
    print("\n激活扩散测试:")
    activation = sn.spread_activation(["canary"], decay=0.8, threshold=0.1)
    for node, act in sorted(activation.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {act:.3f}")
    
    print("\n路径查找:")
    path = sn.find_path("canary", "animal")
    for edge in path:
        print(f"  {edge.source} -> {edge.relation} -> {edge.target}")
    
    print("\n" + "="*60)
    print("知识图谱测试")
    print("="*60)
    
    kg = create_common_knowledge_graph()
    
    print("\n实体类型查询:")
    mammals = kg.get_entities_by_type("Class")
    print(f"  所有类别: {[n.name for n in mammals]}")
    
    print("\n传递性查询:")
    ancestors = kg.transitive_query("Dog", "is_a")
    print(f"  Dog的所有父类: {ancestors}")
    
    print("\n属性继承推理:")
    has_fur = kg.infer_property("Fido", "has_fur")
    warm_blooded = kg.infer_property("Fido", "warm_blooded")
    print(f"  Fido.has_fur: {has_fur}")
    print(f"  Fido.warm_blooded: {warm_blooded}")
    
    print("\n共同祖先:")
    common = kg.get_common_ancestors("Dog", "Cat")
    print(f"  Dog和Cat的共同祖先: {common}")
    
    print("\n" + "="*60)
    print("图嵌入测试 (TransE)")
    print("="*60)
    
    ge = GraphEmbedding(embedding_dim=10, margin=1.0)
    
    entities = ["canary", "bird", "animal", "yellow", "sing"]
    relations = ["is_a", "has_color", "can"]
    triples = [
        ("canary", "is_a", "bird"),
        ("bird", "is_a", "animal"),
        ("canary", "has_color", "yellow"),
        ("canary", "can", "sing")
    ]
    
    ge.train_transe(triples, entities, epochs=50, learning_rate=0.01)
    
    print("\n链接预测:")
    predictions = ge.predict_links("canary", "is_a", entities, top_k=3)
    for entity, score in predictions:
        print(f"  canary -> is_a -> {entity}: score={score:.3f}")
    
    print("\n" + "="*60)
    print("参考文献")
    print("="*60)
    print("Quillian (1968) - 语义记忆奠基论文")
    print("Brachman (1985) - 语义网络知识表示")
    print("Singhal (2012) - Google知识图谱")
    print("Bordes et al. (2013) - TransE图嵌入算法")