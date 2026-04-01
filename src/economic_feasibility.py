# src/economic_feasibility.py

import json
import networkx as nx
import os

# 关节制造成本归一化向量
COST_DICT = {
    'R': 0.1,  # Revolute joint (转动)
    'U': 0.3,  # Universal joint (万向)
    'P': 0.4,  # Prismatic joint (移动)
    'S': 0.6  # Spherical joint (球面)
}


def evaluate_manufacturability(json_filepath: str, c_max: float = 3.0) -> dict:
    """
    计算图语法生成的并联机器人的经济可行性 (Manufacturability M-value)
    :param json_filepath: 机器人的 JSON 拓扑文件路径
    :param c_max: 理论最大成本 (默认 3.0，代表三条支链极其昂贵的极端配置)
    :return: 包含得分、总成本和关节统计的字典
    """
    if not os.path.exists(json_filepath):
        raise FileNotFoundError(f"找不到文件: {json_filepath}")

    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 加载为有向图
    G = nx.node_link_graph(data, directed=True)

    # 1. 统计单条支链的关节数量 (频次 w_i)
    counts_single_branch = {'R': 0, 'U': 0, 'P': 0, 'S': 0}

    for node_id, attrs in G.nodes(data=True):
        j_type = attrs.get('joint_type', '')

        if not j_type:
            continue

        if j_type in ['R1', 'R2', 'R3']:
            counts_single_branch['R'] += 1
        elif j_type.startswith('U'):
            counts_single_branch['U'] += 1
        elif j_type == 'P1':
            counts_single_branch['P'] += 1
        elif j_type == 'S1':
            counts_single_branch['S'] += 1

    # 2. 乘以 3 得到并联机构实际的总关节数量
    total_counts = {k: v * 3 for k, v in counts_single_branch.items()}

    # 3. 计算实际总成本
    actual_total_cost = sum(total_counts[k] * COST_DICT[k] for k in total_counts)

    # 4. 计算 M-value 得分
    # 限制 M 值最小为 0（防范极端生成的超长昂贵支链导致负数）
    m_value = 1.0 - (actual_total_cost / c_max)
    m_value = max(0.0, m_value)

    return {
        "M_score": round(m_value, 4),
        "total_cost": round(actual_total_cost, 4),
        "joint_counts": total_counts,
        "is_valid": actual_total_cost > 0  # 防止空图
    }


# ==========================================
# 独立测试模块
# ==========================================
if __name__ == "__main__":
    # 测试读取一个 JSON
    test_file = "../generated_xml/0fb333d7.json"  # 替换成你实际拥有的文件名
    if os.path.exists(test_file):
        result = evaluate_manufacturability(test_file)
        print("📊 经济可行性分析报告:")
        print(f"  👉 总制造成本: {result['total_cost']}")
        print(f"  👉 经济性得分 (M-value): {result['M_score']} (越靠近1越好)")
        print(f"  👉 关节总计消耗: {result['joint_counts']}")