# src/GGrammar.py
import math

import networkx as nx
import random
import uuid
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict

class FunctionalRuleGrammar:
    """
    基于函数的策略规则系统：
    每个规则都是一个函数，接收(node_id, graph)并返回(new_nodes, new_edges)
    """

    def __init__(self):
        # 规则函数列表：每个函数实现一种"策略"
        self.rule_functions: List[Callable] = [
            self.rule_s_to_ap,  # 策略1: S = E => P
            # self.rule_e_to_e2dlj2lj,
            self.rule_a_to_axj,
            self.rule_x_to_l,
            self.rule_a_to_e,
            # self.rule_d_to_dlj,
            # self.rule_d_to_none  # 策略5: D = None (新增删除规则)
            self.rule_j_to_r1,
            self.rule_j_to_r2,
            self.rule_j_to_r3,
            self.rule_j_to_u1,
            self.rule_j_to_u2,
            self.rule_j_to_u3,
            self.rule_j_to_u4,
            self.rule_j_to_u5,
            self.rule_j_to_u6,
            self.rule_j_to_s1,
            self.rule_j_to_p1
        ]

    def create_initial_graph(self) -> nx.DiGraph:
        """创建初始图：只有起始节点S"""
        G = nx.DiGraph()
        G.add_node(str(uuid.uuid4())[:16], symbol='S')
        return G

    def rule_s_to_ap(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略1将S替换为A=>P"""
        id1 = str(uuid.uuid4())[:16]
        id2 = str(uuid.uuid4())[:16]
        E_offset = 1

        new_nodes = [
            {'id': id1, 'symbol': 'A', 'offset': E_offset},
            {'id': id2, 'symbol': 'P'}
        ]
        new_edges = [
            (id1, id2, {'type': 'chain', 'created_by': 'rule_s_to_ap'})
        ]
        return new_nodes, new_edges

    # def rule_e_to_e2dlj2lj(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
    #     """策略2将E替换为E=>(D=>L=>J)*2=>L=>J"""
    #     # 生成9个新ID
    #     ids = [str(uuid.uuid4())[:16] for _ in range(9)]
    #
    #     new_nodes = [
    #         {'id': ids[0], 'symbol': 'E'},
    #         {'id': ids[1], 'symbol': 'D'},
    #         {'id': ids[2], 'symbol': 'L'},
    #         {'id': ids[3], 'symbol': 'J'},
    #         {'id': ids[4], 'symbol': 'D'},
    #         {'id': ids[5], 'symbol': 'L'},
    #         {'id': ids[6], 'symbol': 'J'},
    #         {'id': ids[7], 'symbol': 'L'},
    #         {'id': ids[8], 'symbol': 'J'}
    #     ]
    #
    #     tag = 'rule_e_to_e2dlj2lj'
    #     new_edges = [
    #         (ids[0], ids[1], {'type': 'chain', 'created_by': tag}),
    #         (ids[1], ids[2], {'type': 'chain', 'created_by': tag}),
    #         (ids[2], ids[3], {'type': 'chain', 'created_by': tag}),
    #         (ids[3], ids[7], {'type': 'chain', 'created_by': tag}),  # 注意原代码逻辑连接
    #         (ids[0], ids[4], {'type': 'chain', 'created_by': tag}),
    #         (ids[4], ids[5], {'type': 'chain', 'created_by': tag}),
    #         (ids[5], ids[6], {'type': 'chain', 'created_by': tag}),
    #         (ids[6], ids[7], {'type': 'chain', 'created_by': tag}),
    #         (ids[7], ids[8], {'type': 'chain', 'created_by': tag})
    #     ]
    #
    #     return new_nodes, new_edges

    def rule_a_to_axj(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略3将A替换为A=>X=>J"""
        id1 = str(uuid.uuid4())[:16]
        id2 = str(uuid.uuid4())[:16]
        id3 = str(uuid.uuid4())[:16]
        # E_offset = 1
        # L_length = 1

        new_nodes = [
            {'id': id1, 'symbol': 'A'},
            {'id': id2, 'symbol': 'X'},
            {'id': id3, 'symbol': 'J'}
        ]
        new_edges = [
            (id1, id2, {'type': 'chain', 'created_by': 'rule_a_to_axj'}),
            (id2, id3, {'type': 'chain', 'created_by': 'rule_a_to_axj'})
        ]
        return new_nodes, new_edges

    # def rule_d_to_dlj(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
    #     """策略4将D替换为D=>L=>J"""
    #     id1 = str(uuid.uuid4())[:16]
    #     id2 = str(uuid.uuid4())[:16]
    #     id3 = str(uuid.uuid4())[:16]
    #
    #     new_nodes = [
    #         {'id': id1, 'symbol': 'D'},
    #         {'id': id2, 'symbol': 'L'},
    #         {'id': id3, 'symbol': 'J'}
    #     ]
    #     new_edges = [
    #         (id1, id2, {'type': 'chain', 'created_by': 'rule_e_to_dlj'}),
    #         (id2, id3, {'type': 'chain', 'created_by': 'rule_e_to_dlj'})
    #     ]
    #     return new_nodes, new_edges

    # def rule_d_to_none(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
    #     """策略5: D → None (删除D节点及其所有边)"""
    #     return [], []

    def rule_x_to_l(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 X 替换为一个全新的 L 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]
        L_length = random.uniform(0.5, 1)

        # 2. 完全重新定义新节点 L 的属性，赋予全新的初始值
        new_nodes = [
            {'id': id1, 'symbol': 'L', 'length': L_length}
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_a_to_e(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 A 替换为一个全新的 E 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]
        L_length = random.uniform(0.5, 1)

        # 2. 完全重新定义新节点 L 的属性，赋予全新的初始值
        new_nodes = [
            {'id': id1, 'symbol': 'L', 'length': L_length}
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_r1(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 R 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 R 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'R1',
                'theta_int': random.uniform(-math.pi, math.pi),
                'theta_r': math.pi,
                'joint_type': 'R1'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_r2(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 R 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 R 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'R2',
                'theta_int': random.uniform(-math.pi, math.pi),
                'theta_r': math.pi,
                'joint_type': 'R2'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_r3(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 R 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 R 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'R3',
                'theta_int': random.uniform(-math.pi, math.pi),
                'theta_r': math.pi,
                'joint_type': 'R3'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_u1(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 U 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 U 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'U1',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'joint_type': 'U1'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_u2(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 U 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 U 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'U2',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'joint_type': 'U2'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_u3(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 U 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 U 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'U3',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'joint_type': 'U3'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_u4(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 U 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 U 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'U4',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'joint_type': 'U4'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_u5(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 U 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 U 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'U5',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'joint_type': 'U5'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_u6(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 U 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 U 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'U6',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'joint_type': 'U6'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_s1(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 S 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 S 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'S1',
                'theta_int1': random.uniform(-math.pi, math.pi),
                'theta_int2': random.uniform(-math.pi, math.pi),
                'theta_int3': random.uniform(-math.pi, math.pi),
                'theta_r1': math.pi,
                'theta_r2': math.pi,
                'theta_r3': math.pi,
                'joint_type': 'S1'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def rule_j_to_p1(self, node_id: str, G: nx.DiGraph) -> Tuple[List[Dict], List[Tuple[str, str, Dict]]]:
        """策略: 将 J 替换为一个全新的 P 节点"""
        # 1. 生成新节点的 ID
        id1 = str(uuid.uuid4())[:16]

        # 2. 完全重新定义新节点 P 的属性，赋予全新的初始值
        new_nodes = [
            {
                'id': id1,
                'symbol': 'P1',
                'theta_int': random.uniform(-math.pi, math.pi),
                'theta_r': math.pi,
                'joint_type': 'P1'
            }
        ]

        # 3. 1对1的单节点替换，不需要创建任何内部连接边
        # 外围输入/输出边的重新连接会由核心方法 apply_functional_rule 自动处理
        new_edges = []

        return new_nodes, new_edges

    def apply_functional_rule(self, G: nx.DiGraph, node_id: str, rule_func: Callable) -> nx.DiGraph:
        """应用函数式规则：核心方法"""
        new_G = G.copy()
        new_nodes, new_edges = rule_func(node_id, new_G)

        # 判断是否为删除操作
        if not new_nodes:
            # 删除模式：桥接或移除边
            in_edges = list(new_G.in_edges(node_id, data=True))
            out_edges = list(new_G.out_edges(node_id, data=True))

            for src, _, in_data in in_edges:
                for _, dst, out_data in out_edges:
                    merged_data = {**out_data, **in_data, 'bridged': True}
                    new_G.add_edge(src, dst, **merged_data)

            new_G.remove_node(node_id)
            return new_G

        new_node_ids = [node_data.pop('id') for node_data in new_nodes]

        for i, node_data in enumerate(new_nodes):
            new_G.add_node(new_node_ids[i], **node_data)

        in_edges = list(new_G.in_edges(node_id, data=True))
        if in_edges and new_node_ids:
            first_node_id = new_node_ids[0]
            for src, _, data in in_edges:
                new_G.add_edge(src, first_node_id, **data)

        out_edges = list(new_G.out_edges(node_id, data=True))
        if out_edges and new_node_ids:
            last_node_id = new_node_ids[-1]
            for _, dst, data in out_edges:
                new_G.add_edge(last_node_id, dst, **data)

        for src, dst, data in new_edges:
            new_G.add_edge(src, dst, **data)

        new_G.remove_node(node_id)
        return new_G

    def get_applicable_rules(self, node_symbol: str) -> List[Callable]:
        """根据节点符号获取可应用的规则函数"""
        applicable = []
        if node_symbol == 'S':
            applicable.append(self.rule_s_to_ep)
        elif node_symbol == 'E':
            applicable.append(self.rule_e_to_exj)
            # applicable.append(self.rule_e_to_e2dlj2lj)
        # elif node_symbol == 'D':
        #     applicable.append(self.rule_d_to_dlj)
        #     applicable.append(self.rule_d_to_none)
        return applicable

    def step(self, G: nx.DiGraph) -> nx.DiGraph:
        """执行一次推导步骤"""
        expandable_nodes = []
        for node_id, node_data in G.nodes(data=True):
            symbol = node_data['symbol']
            applicable_rules = self.get_applicable_rules(symbol)
            if applicable_rules:
                expandable_nodes.append((node_id, symbol, applicable_rules))

        if not expandable_nodes:
            return G

        node_id, node_symbol, applicable_rules = random.choice(expandable_nodes)
        rule_func = random.choice(applicable_rules)

        # print('Applied', node_symbol,'(uuid:', node_id, ')', rule_func.__name__)
        new_G = self.apply_functional_rule(G, node_id, rule_func)
        return new_G

    def finalize_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        最终阶段：从图的最开始依次扫描，将所有的泛型节点 X 替换为连杆 L，
        并将泛型关节 J 替换为具体的关节节点。
        """
        new_G = G.copy()

        # 1. 对图进行拓扑排序，确保从前到后（即从图的起始端到末端）的顺序
        try:
            topo_order = list(nx.topological_sort(new_G))
        except nx.NetworkXUnfeasible:
            # 如果图中存在环路（虽然在这个文法中通常不会），则退化为默认顺序
            topo_order = list(new_G.nodes())

        # 2. 按照拓扑顺序，收集所有需要被替换的 'X' 和 'J' 节点的 ID
        target_nodes = [node_id for node_id in topo_order if new_G.nodes[node_id].get('symbol') in ('X', 'J')]

        # 3. 定义所有用于替换 J 的具体规则列表 (保持原有逻辑不变)
        j_replacement_rules = [
            self.rule_j_to_r1, self.rule_j_to_r2, self.rule_j_to_r3,
            self.rule_j_to_u1, self.rule_j_to_u2, self.rule_j_to_u3,
            self.rule_j_to_u4, self.rule_j_to_u5, self.rule_j_to_u6,
            self.rule_j_to_s1,
            self.rule_j_to_p1
        ]

        # 4. 定义 J 替换对应的权重列表 (保持原有逻辑不变)
        j_weights = [
            20, 20, 20,  # R1, R2, R3 (转动关节最常用，权重给高点，共60)
            3, 3, 3,  # U1, U2, U3 (万向关节相对较少)
            3, 3, 3,  # U4, U5, U6 (共18)
            10,  # S1 (球面关节)
            12  # P1 (移动关节)
        ]

        # 5. 遍历排序好的目标节点列表，依次应用对应的替换规则
        for node_id in target_nodes:
            symbol = new_G.nodes[node_id].get('symbol')

            if symbol == 'X':
                # 应用 X 替换为 L 的规则
                rule_func = self.rule_x_to_l
                print(f"Finalizing X node (uuid: {node_id}) -> {rule_func.__name__}")
                new_G = self.apply_functional_rule(new_G, node_id, rule_func)

            elif symbol == 'J':
                # 应用 J 替换为具体关节的规则（按权重随机抽取）
                rule_func = random.choices(j_replacement_rules, weights=j_weights, k=1)[0]
                print(f"Finalizing J node (uuid: {node_id}) -> {rule_func.__name__}")
                new_G = self.apply_functional_rule(new_G, node_id, rule_func)

        return new_G


def visualize_functional(G: nx.DiGraph):
    """
    从左到右的可视化
    """
    plt.figure(figsize=(14, 5))

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo_order = sorted(G.nodes())

    pos = {node: [i, 0] for i, node in enumerate(topo_order)}

    # [修改] 提取符号的第一个字母用于排版偏移计算（比如把 'R1' 当作 'R' 处理）
    symbol_y_offset = {'S': 0.0, 'P': 0.0, 'J': 0.15, 'L': -0.15, 'E': 0.0, 'D': 0.0, 'R': 0.15, 'U': 0.15}
    symbol_count = {}

    for node in topo_order:
        symbol = G.nodes[node]['symbol']
        base_char = symbol[0] # 取首字母 R, U, S, P
        offset = symbol_y_offset.get(base_char, 0) + symbol_count.get(symbol, 0) * 0.02
        pos[node][1] = offset
        symbol_count[symbol] = symbol_count.get(symbol, 0) + 1

    # [修改] 为不同类型的关节分配不同色系的颜色
    def get_color(sym):
        if sym.startswith('R'): return '#FFE4B5' # 转动关节：浅黄
        if sym.startswith('U'): return '#87CEFA' # 万向关节：天蓝
        if sym.startswith('S'): return '#DDA0DD' # 球面关节：梅红
        if sym.startswith('P'): return '#98FB98' # 移动/动平台：浅绿
        colors = {'S': '#FF6B6B', 'J': '#95E77E', 'L': '#FFA07A', 'E': '#DDA0DD'}
        return colors.get(sym, '#C0C0C0')

    node_color = [get_color(G.nodes[n]['symbol']) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_color,
                           node_size=1500, alpha=0.85, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=25,
                           edge_color='gray', width=2, alpha=0.7)

    labels = {n: f"{G.nodes[n]['symbol']}\n{n.split('_')[-1][:4]}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

    plt.title(f"Visualization: all {len(G.nodes())} nodes", fontsize=14, pad=20)
    plt.ylim(-0.5, 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()