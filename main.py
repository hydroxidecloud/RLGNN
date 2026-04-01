# main.py
import os
import json
import uuid
import torch
import random
import networkx as nx
import mujoco
import concurrent.futures
import shutil
from tinydb import TinyDB, Query
from torch_geometric.data import Batch
import torch.nn.functional as F
import torch_geometric.transforms as T  # 引入转换器
import hashlib
import matplotlib.pyplot as plt
import csv

# 导入你写的模块
from src.GGrammar import FunctionalRuleGrammar
# from src.gnn_model import SimpleTopologyPredictor
from Net import Net
from src.gragh2xml import graph_to_mujoco_xml, save_mujoco_xml

# 【引入最新分离出来的核心仿真与入库模块】
from src.simulator import evaluate_robot_headless, save_simulation_to_tinydb
from src.calc_screw_dof import compute_screw_theory_dof
# from src.trainer import train_gnn

MAX_NODES = 50

# ==========================================
# 🧠 新增：启发式状态经验池 (StatesPool)
# ==========================================
class SimpleStatesPool:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        # 字典结构：图的哈希值 -> (PyG稠密张量, 最高目标潜力分)
        self.pool = {}

    def push(self, g_hash, pyg_data, target_value):
        if g_hash in self.pool:
            # 核心魔法：如果以前见过这个半成品，保留它历史跑出的【最高分】
            self.pool[g_hash] = (pyg_data, max(self.pool[g_hash][1], target_value))
        else:
            if len(self.pool) >= self.capacity:
                del self.pool[next(iter(self.pool))] # 满了就删掉最旧的
            self.pool[g_hash] = (pyg_data, target_value)

    def sample(self, batch_size):
        return random.sample(list(self.pool.values()), min(len(self.pool), batch_size))

    def __len__(self):
        return len(self.pool)


# ==========================================
# 1. 内存级数据转换器 (19D 独热稠密图版)
# ==========================================
def graph_to_pyg_data(G):
    """将内存中的 NetworkX 图转换为 GNN (DiffPool) 能看懂的 19D 稠密张量"""
    type_map = {'E': 0, 'L': 1, 'P': 2, 'R1': 3, 'R2': 4, 'R3': 5, 'P1': 6,
                'U1': 7, 'U2': 7, 'U3': 7, 'U4': 7, 'U5': 7, 'U6': 7, 'S1': 8,
                'X': 9, 'J': 10}

    node_features = []
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    for node in G.nodes():
        n_data = G.nodes[node]
        symbol = n_data.get('symbol', 'E')
        j_type = n_data.get('joint_type', symbol)
        type_idx = int(type_map.get(j_type, type_map.get(symbol, 0)))

        # ⭐ 修改 1：独热编码 (One-Hot) 彻底解决数字带来的偏序误导 (11维)
        one_hot_type = [0.0] * 11
        one_hot_type[type_idx] = 1.0

        # ⭐ 修改 2：保持连续物理属性 (8维)
        feat_length = float(n_data.get('length', 0.0))
        feat_offset = float(n_data.get('offset', 0.0))
        feat_t_int1 = float(n_data.get('theta_int', n_data.get('theta_int1', 0.0)))
        feat_t_int2 = float(n_data.get('theta_int2', 0.0))
        feat_t_int3 = float(n_data.get('theta_int3', 0.0))
        feat_t_r1 = float(n_data.get('theta_r', n_data.get('theta_r1', 0.0)))
        feat_t_r2 = float(n_data.get('theta_r2', 0.0))
        feat_t_r3 = float(n_data.get('theta_r3', 0.0))

        # 拼接成最终的 19 维特征
        n_features = one_hot_type + [
            feat_length, feat_offset,
            feat_t_int1, feat_t_int2, feat_t_int3,
            feat_t_r1, feat_t_r2, feat_t_r3
        ]
        node_features.append(n_features)

    from torch_geometric.data import Data
    x = torch.tensor(node_features, dtype=torch.float)
    edges = list(G.edges())

    if len(edges) > 0:
        edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in edges],
                                  dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 先构建普通的稀疏图数据
    sparse_data = Data(x=x, edge_index=edge_index)

    # ⭐ 修改 3：将其强行填充、转换为固定大小的稠密矩阵
    dense_transform = T.ToDense(num_nodes=MAX_NODES)
    dense_data = dense_transform(sparse_data)

    # 此时 dense_data 将拥有 .x, .adj, .mask 三个属性
    return dense_data


# ==========================================
# 2. 核心：ε-贪心规则选择器 (平行宇宙 DiffPool 版)
# ==========================================
def epsilon_greedy_step(current_G, target_node, valid_rules, epsilon, model, device, grammar, num_universes=10,
                        verbose=False):
    candidates = []

    for _ in range(num_universes):
        temp_G = current_G.copy()
        rule_func = random.choice(valid_rules)
        new_G = grammar.apply_functional_rule(temp_G, target_node, rule_func)
        candidates.append({'graph': new_G, 'rule_name': rule_func.__name__})

    if random.random() < epsilon:
        # 探索
        chosen = random.choice(candidates)
        if verbose: print(f"    🎲 [探索] 选中: {chosen['rule_name']}")
        return chosen['graph']
    else:
        # ==========================================
        # ⚡ GNN 极速并行打分 (DiffPool 稠密批处理)
        # ==========================================
        model.eval()

        # 1. 把 10 个平行宇宙的图纸先转成 PyG 稠密格式
        data_list = [graph_to_pyg_data(cand['graph']) for cand in candidates]

        # 2. ⭐ 【核心魔法】：直接沿着 batch 维度堆叠张量
        batch_x = torch.stack([d.x for d in data_list]).to(device)  # [10, 50, 19]
        batch_adj = torch.stack([d.adj for d in data_list]).to(device)  # [10, 50, 50]
        batch_mask = torch.stack([d.mask for d in data_list]).to(device)  # [10, 50]

        with torch.no_grad():
            # 3. GPU 火力全开：传入特征、邻接矩阵和掩码
            # 注意：DiffPool 前向传播返回三个值 (output, link_loss, ent_loss)
            # 在预测/打分阶段，我们果断抛弃后面的辅助 loss，只要 [0]
            scores_tensor, _, _ = model(batch_x, batch_adj, batch_mask)

            # scores_tensor 的形状是 [10, 1]，将它展平为 [10] 的一维数组
            scores = scores_tensor.view(-1)

        # 4. 找出分数最高的那一个宇宙的索引
        best_idx = torch.argmax(scores).item()

        if verbose:
            print(
                f"    🧠 [GNN 大脑] 看好并选中了: {candidates[best_idx]['rule_name']} (预测分: {scores[best_idx].item():.2f})")

        return candidates[best_idx]['graph']


# ==========================================
# 3. 单个机器人生成流水线 (记录进化史版)
# ==========================================
def generate_single_robot(model, device, grammar, step=4, epsilon=0.2, num_universes=10):
    G = grammar.create_initial_graph()
    history = []  # <--- 新增：用于记录图纸的进化过程

    # 宏观推导阶段
    for _ in range(step):
        G = grammar.step(G)
        history.append(G.copy())  # 记录每一个宏观半成品

    valid_rules_for_x = [grammar.rule_x_to_l]
    valid_rules_for_j = [
        grammar.rule_j_to_r1, grammar.rule_j_to_r2, grammar.rule_j_to_r3,
        grammar.rule_j_to_u1, grammar.rule_j_to_u2, grammar.rule_j_to_u3,
        grammar.rule_j_to_u4, grammar.rule_j_to_u5, grammar.rule_j_to_u6,
        grammar.rule_j_to_s1, grammar.rule_j_to_p1
    ]

    # 微观实体化阶段
    while True:
        target_nodes = [n for n in G.nodes() if G.nodes[n].get('symbol') in ('X', 'J')]
        if not target_nodes:
            break

        node_id = random.choice(target_nodes)
        symbol = G.nodes[node_id].get('symbol')

        if symbol == 'X':
            G = epsilon_greedy_step(G, node_id, valid_rules_for_x, epsilon, model, device, grammar, num_universes,
                                    verbose=False)
        elif symbol == 'J':
            G = epsilon_greedy_step(G, node_id, valid_rules_for_j, epsilon, model, device, grammar, num_universes,
                                    verbose=False)

        history.append(G.copy())  # <--- 新增：记录每一次坍缩后的半成品

    # 返回整个进化史列表，列表的最后一个元素就是完全体
    return history


# ==========================================
# 4. 批量生成主程序 (支持 Epoch 多轮进化)
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print(" 🏭 GHS 机器人批量设计工厂启动 (启发式在线进化版) ")
    print("=" * 50)

    # 【核心参数设置】
    NUM_EPOCHS = 100  # 你想要运行多少轮 (Epoch)
    NUM_ROBOTS = 256  # 每一轮生成多少台机器人
    MACRO_STEPS = 4  # 图文法宏观推导步数
    EPSILON = 0.2  # 探索概率
    NUM_UNIVERSES = 100  # 每次挑选零件时的“平行宇宙”数量

    # 【启发式训练超参数】
    TRAIN_BATCH_SIZE = 64  # 每次从经验池抓取的图纸数量
    OPT_ITERS = 20  # 每轮跑完后，大脑反思/训练的次数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化你的层级池化大脑
    model = Net(max_nodes=MAX_NODES, num_channels=19, num_outputs=1).to(device)

    # 初始化优化器和经验池
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    weight_path = 'weights.pth'

    grammar = FunctionalRuleGrammar()
    total_success = 0

    # ==========================================
    # 📊 统计模块初始化
    # ==========================================
    stats_epochs = []
    stats_avg_scores = []
    stats_avg_scores_all = []
    stats_max_scores = []
    stats_valid_counts = []
    stats_losses = []

    global_elites = [] # 记录所有世代的顶尖机器人

    # 在循环开始前，新建 CSV 文件并写入表头 (模式为 'w'，覆盖旧文件)
    csv_path = os.path.join("data", "evolution_stats.csv")
    os.makedirs("data", exist_ok=True)  # 确保 data 文件夹存在
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Score_NonZero', 'Avg_Score_All', 'Max_Score', 'Valid_Robots', 'Loss'])

    for epoch in range(1, NUM_EPOCHS + 1):
        states_pool = SimpleStatesPool(capacity=50000)
        print("\n" + "★" * 50)
        print(f" 🌀 开始执行 Epoch {epoch}/{NUM_EPOCHS} ")
        print("★" * 50)

        # 加载历史权重
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            print(f"  ✅ [Epoch {epoch}] 成功加载最新 GNN 权重！")
        else:
            print(f"  ⚠️ [Epoch {epoch}] 未找到权重，GNN 使用随机直觉。")

        # 动态创建属于当前 Epoch 的专属文件夹
        epoch_base_dir = os.path.join("data", f"epoch_{epoch}")
        epoch_json_dir = os.path.join(epoch_base_dir, "generated_xml")
        epoch_xml_dir = os.path.join(epoch_base_dir, "generated_xml")
        os.makedirs(epoch_json_dir, exist_ok=True)
        os.makedirs(epoch_xml_dir, exist_ok=True)

        print(f"  📁 本轮 JSON 保存至: ./{epoch_json_dir}/")
        print(f"  📁 本轮 XML  保存至: ./{epoch_xml_dir}/\n")

        # ==========================================
        # 🚀 阶段 1：【串行生成与安检】极速产出图纸 (记录完整轨迹)
        # ==========================================
        print(f"  🤖 正在生成 {NUM_ROBOTS} 台机器人的图纸并进行安检...")
        epoch_success_count = 0

        pending_simulations = []  # 存放送去并行仿真的任务
        results_data = []  # 存放所有机器人的最终数据

        for i in range(NUM_ROBOTS):
            robot_id = uuid.uuid4().hex
            try:
                # ==========================================
                # 🌟 新增：记录这台机器人的完整成长轨迹
                # 注：你需要将 generate_single_robot 的内部逻辑也调整为
                # 记录每一次 epsilon_greedy_step 后的状态。这里假设 history_graphs
                # 已经包含了从初始图到最终图的所有中间状态。
                # ==========================================
                history_graphs = generate_single_robot(model, device, grammar, step=MACRO_STEPS, epsilon=EPSILON,
                                                       num_universes=NUM_UNIVERSES)
                G_final = history_graphs[-1]

                # trajectory 就是这台机器人的生命轨迹
                trajectory = history_graphs.copy()

                # 转为 XML
                xml_path = os.path.join(epoch_xml_dir, f"{robot_id}.xml")
                xml_content = graph_to_mujoco_xml(G_final)
                save_mujoco_xml(xml_content, xml_path)

                # 运动学预检 (Fail-Fast)
                m_test = mujoco.MjModel.from_xml_path(xml_path)
                if m_test.nu == 0:
                    results_data.append((robot_id, 0.0, history_graphs, "❌ 无驱动器"))
                else:
                    dof = compute_screw_theory_dof(xml_path, run_sim=False)
                    if dof <= 0:
                        results_data.append((robot_id, 0.0, history_graphs, f"❌ 死锁 (DOF={dof})"))
                    elif dof > 3:
                        results_data.append((robot_id, 0.0, history_graphs, f"❌ 散架 (DOF={dof})"))
                    else:
                        # 🌟 修改：把 trajectory 也一起传给后续仿真，方便发奖励！
                        pending_simulations.append((robot_id, xml_path, history_graphs, trajectory))

            except Exception as e:
                results_data.append((robot_id, 0.0, [], f"❌ 生成崩溃: {e}"))

        # ==========================================
        # 🚀 阶段 2：【并行仿真】压榨算力 & 目标值回传 (Target Value Backprop)
        # ==========================================
        if pending_simulations:
            print(f"  ⚡ 预检完成，共有 {len(pending_simulations)} 台机构进入多核并行仿真...")
            max_workers = os.cpu_count() or 4

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 🌟 修改：传递的 task 现在包含 trajectory
                future_to_robot = {
                    executor.submit(evaluate_robot_headless, task[1]): task
                    for task in pending_simulations
                }

                for future in concurrent.futures.as_completed(future_to_robot):
                    # 🌟 提取 trajectory
                    robot_id, xml_path, history_graphs, trajectory = future_to_robot[future]
                    try:
                        perf_score = future.result()

                        if perf_score > 0.0:
                            results_data.append((robot_id, perf_score, history_graphs, "✅ 仿真有效"))
                            global_elites.append((perf_score, robot_id, xml_path))
                            epoch_success_count += 1
                            total_success += 1

                            import hashlib  # 如果文件开头没引，这里或者顶部引入一下

                            # ==========================================
                            # 🌟🌟🌟 核心魔法：目标值回溯 (Target Value Backpropagation)
                            # ==========================================
                            # 将这台机器人的最终得分，赋给它一路上的所有半成品祖先！
                            for state_G in trajectory:
                                # 1. 转成 19D 稠密张量
                                pyg_state = graph_to_pyg_data(state_G)

                                # 2. 【修复】直接用 GNN 视角的特征张量和邻接矩阵生成 MD5 绝对哈希！
                                # 这样只要零件种类、连续物理参数或图结构有任何不同，都会被视为全新状态
                                state_bytes = pyg_state.x.cpu().numpy().tobytes() + pyg_state.adj.cpu().numpy().tobytes()
                                g_hash = hashlib.md5(state_bytes).hexdigest()

                                # 3. 压入经验池
                                states_pool.push(g_hash, pyg_state, perf_score)

                        else:
                            results_data.append((robot_id, 0.0, history_graphs, "❌ 表现无效 (得分0.0)"))

                    except Exception as e:
                        results_data.append((robot_id, 0.0, history_graphs, f"❌ 仿真崩溃: {e}"))

        # ==========================================
        # 🚀 阶段 3：【输出记录】整理并打印结果
        # ==========================================
        print(f"  💾 正在打印本轮产出状态...")
        epoch_valid_scores = []  # 用于收集本轮所有有效得分
        epoch_all_scores = []
        for i, (robot_id, perf_score, history_graphs, status) in enumerate(results_data):
            epoch_all_scores.append(perf_score) # 🌟 新增：无论死活，先把分数记下来（失败的机器人在前面已经被设为 0.0 了）
            if "✅" in status:
                print(f"    [{i + 1}/{NUM_ROBOTS}] 🤖 {robot_id[:8]}... | 得分: {perf_score:.4f} | {status}")
                # （可选：如果你还需要把 json 存到磁盘，可以在这里继续写你的 json.dump 逻辑）
                if perf_score > 0.0:
                    epoch_valid_scores.append(perf_score)
            else:
                print(f"    [{i + 1}/{NUM_ROBOTS}] 🗑️ {robot_id[:8]}... | 得分: 0.0000 | {status}")

        print(f"  🏁 Epoch {epoch} 结束，成功产出 {epoch_success_count} 台有效机器人。")

        # 🌟🌟🌟 统计两种平均分
        # 1. 非零平均分 (剔除废铁后的平均水平)
        current_avg = sum(epoch_valid_scores) / len(epoch_valid_scores) if epoch_valid_scores else 0.0
        # 2. 🌟 新增：含零平均分 (所有产出的综合期望)
        current_avg_all = sum(epoch_all_scores) / len(epoch_all_scores) if epoch_all_scores else 0.0

        current_max = max(epoch_valid_scores) if epoch_valid_scores else 0.0

        stats_epochs.append(epoch)
        stats_avg_scores.append(current_avg)
        stats_avg_scores_all.append(current_avg_all)
        stats_max_scores.append(current_max)
        stats_valid_counts.append(epoch_success_count)

        # ==========================================
        # ⭐ Phase 4 -> 🧠 大脑在线进化 (Learn & Update)
        # ==========================================
        print(f"\n  🧠 开始经验回放与大脑进化 (当前经验池容量: {len(states_pool)} 个半成品状态)...")
        epoch_loss = 0.0  # 默认本轮 Loss 为 0

        if len(states_pool) >= 4:
            model.train()
            total_loss = 0.0

            # 🌟 动态计算本轮的 Batch Size
            current_batch_size = min(TRAIN_BATCH_SIZE, len(states_pool))

            for opt_step in range(OPT_ITERS):
                # 1. 从经验池中随机抽取历史半成品和它们对应的最高潜力分
                samples = states_pool.sample(current_batch_size)

                # 2. 像叠千层饼一样，把它们堆叠成 Batch 张量
                batch_x = torch.stack([s[0].x for s in samples]).to(device)
                batch_adj = torch.stack([s[0].adj for s in samples]).to(device)
                batch_mask = torch.stack([s[0].mask for s in samples]).to(device)

                # 真实的 Target Value (最高潜力分)
                batch_y = torch.tensor([s[1] for s in samples], dtype=torch.float32).to(device)

                # 3. 标准的 PyTorch 训练流
                optimizer.zero_grad()

                # DiffPool 输出预测分以及两个辅助池化 Loss
                predictions, link_loss, ent_loss = model(batch_x, batch_adj, batch_mask)

                # 计算预测分与真实潜力分之间的均方误差
                mse_loss = F.mse_loss(predictions.view(-1), batch_y)

                # 🌟 DiffPool 必须加上辅助 Loss！
                loss = mse_loss + link_loss + ent_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / OPT_ITERS
            print(f"  📈 训练完成！本轮更新 {OPT_ITERS} 次，平均 Loss: {epoch_loss:.4f}")

            # 🌟 训练完立刻保存权重！
            torch.save(model.state_dict(), weight_path)
            print(f"  💾 权重已实时保存至: {weight_path}")
        else:
            print("  ⏳ 经验池数据不足，继续积累数据...")

        # 将本轮的 Loss 添加到全局统计中
        stats_losses.append(epoch_loss)

        # 🌟🌟🌟 新增：实时将本轮收集到的最新数据追加写入 CSV (模式为 'a'，追加)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                current_avg,
                current_avg_all,
                current_max,
                epoch_success_count,
                epoch_loss
            ])
        print(f"  📝 本轮统计数据已实时追加至: {csv_path}")

        # ==========================================
        # 🏆 最后一轮精英作品提取库 (Hall of Fame)
        # ==========================================
        if epoch == NUM_EPOCHS:
            print(f"\n  🌟 正在提取横跨全纪元的 Top 50 精英作品...")
            final_xml_dir = os.path.join("data", "epoch_final")
            os.makedirs(final_xml_dir, exist_ok=True)

            # 🌟 按得分从高到低排序，只取历史最高的前 50 台
            global_elites.sort(key=lambda x: x[0], reverse=True)
            top_robots = global_elites[:50]

            for rank, (score, r_id, original_xml) in enumerate(top_robots):
                target_xml = os.path.join(final_xml_dir, f"Rank{rank + 1}_Score{score:.2f}_{r_id}.xml")
                if os.path.exists(original_xml):
                    shutil.copy(original_xml, target_xml)

            print(f"  ✅ 完美收官！已将历史最强的 {len(top_robots)} 台机器人归档。")

    print("\n" + "=" * 50)
    print(f"🎉 全部生成任务圆满结束！总计产出 {total_success} 台机器人。")
    print("=" * 50)

    # ==========================================
    # 📈 阶段 5：绘制并导出全局统计图表与数据
    # ==========================================
    print("\n  📊 正在生成全局进化统计图表...")

    # 🌟 修改：画布加宽，变成宽 16，以容纳 3 个图表
    plt.figure(figsize=(16, 5))

    # 子图 1: 得分趋势 (平均分 & 最高分)
    plt.subplot(1, 3, 1)

    plt.plot(stats_epochs, stats_avg_scores, label='Avg Score (Non-zero)', color='#1f77b4', linewidth=2)
    plt.plot(stats_epochs, stats_avg_scores_all, label='Avg Score (All inc. 0)', color='#9467bd', linestyle='-.',
             linewidth=2)  # 🌟 新增：含零平均分 (紫色虚点线)
    plt.plot(stats_epochs, stats_max_scores, label='Max Score', color='#ff7f0e', linestyle='--', linewidth=2)

    plt.title('Evolution of Robot Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 子图 2: 有效机器人数量趋势 (百分比)
    plt.subplot(1, 3, 2)  # 1行3列的第2个

    # 🌟 新增：在这里实时将“存活数量”换算为“存活百分比”
    # NUM_ROBOTS 是你在代码顶部定义的每轮总生成数量 (256)
    survival_rates = [(count / NUM_ROBOTS) * 100.0 for count in stats_valid_counts]

    # 🌟 修改：画图时传入换算好的 survival_rates，并修改 label
    plt.plot(stats_epochs, survival_rates, label='Survival Rate', color='#2ca02c', linewidth=2)
    plt.fill_between(stats_epochs, survival_rates, color='#2ca02c', alpha=0.2)

    plt.title('Survival Rate (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Rate (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 🌟 新增：子图 3: Training Loss 趋势
    plt.subplot(1, 3, 3)
    plt.plot(stats_epochs, stats_losses, label='GNN Training Loss', color='#d62728', linewidth=2)
    plt.title('Brain Evolution (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 自动调整布局并保存图片
    plt.tight_layout()
    stats_fig_path = os.path.join("data", "evolution_stats.png")
    plt.savefig(stats_fig_path, dpi=300)
    print(f"  ✅ 统计图表已保存至: {stats_fig_path}")

    # # 同时导出一份 CSV 表格供后续分析
    # csv_path = os.path.join("data", "evolution_stats.csv")
    # with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     # 🌟 修改表头，加上 'Avg_Score_All'
    #     writer.writerow(['Epoch', 'Avg_Score_NonZero', 'Avg_Score_All', 'Max_Score', 'Valid_Robots', 'Loss'])
    #     for i in range(len(stats_epochs)):
    #         # 🌟 写入时多加一个 stats_avg_scores_all[i]
    #         writer.writerow([
    #             stats_epochs[i],
    #             stats_avg_scores[i],
    #             stats_avg_scores_all[i],  # 🌟 新增的这一列
    #             stats_max_scores[i],
    #             stats_valid_counts[i],
    #             stats_losses[i]
    #         ])
    # print(f"  ✅ 详细统计数据已保存至: {csv_path}")