# analyze.py
# 用于遍历所有的 XML 模型，检查驱动器数量并利用螺旋理论计算 DOF，
# 最终统计出“正常”的机构（存在驱动器，且 DOF 在 1~6 之间）。
import os
import sys
import json
import networkx as nx
import mujoco
from contextlib import redirect_stdout
from src.calc_screw_dof import compute_screw_theory_dof
from src.economic_feasibility import evaluate_manufacturability


def analyze_all_robots(xml_dir="generated_xml"):
    """
    遍历所有的 XML 模型，检查驱动器数量并利用螺旋理论计算 DOF，
    最终统计出“正常”的机构（存在驱动器，且 DOF 在 1~6 之间）。
    """
    if not os.path.exists(xml_dir):
        print(f"❌ 找不到目录 '{xml_dir}'！请先运行 main.py 生成模型。")
        return

    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
    total_files = len(xml_files)

    if total_files == 0:
        print(f"⚠️ 目录 '{xml_dir}' 中没有发现任何 XML 文件。")
        return

    print(f"🔍 开始批量分析 {total_files} 个机器人模型 (这可能需要几秒钟)...\n")

    valid_robots = []
    stats = {
        "locked": 0,  # DOF == 0 (死锁/过约束)
        "under_constrained": 0,  # DOF > 6 (欠约束/自由散漫)
        "no_actuator": 0,  # 驱动器数量为 0
        "error": 0  # 加载或计算报错
    }

    for i, filename in enumerate(xml_files):
        xml_path = os.path.join(xml_dir, filename)

        # 终端进度提示 (\r 让其在同一行刷新，保持整洁)
        print(f"⏳ 进度: [{i + 1}/{total_files}] 正在分析: {filename} ...", end="\r")

        try:
            # 1. 快速检查是否有驱动器
            model = mujoco.MjModel.from_xml_path(xml_path)
            if model.nu == 0:
                stats["no_actuator"] += 1
                continue

            # 2. 计算 DOF
            # 使用 os.devnull 屏蔽底层计算时的长篇打印输出
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    # 调用 src 里的计算函数，并关闭弹窗演示 (run_sim=False)
                    dof = compute_screw_theory_dof(xml_path, run_sim=False)

            # 3. 分类统计
            if dof == -1:
                stats["error"] += 1
            elif dof == 0:
                stats["locked"] += 1
            elif dof > 6:
                stats["under_constrained"] += 1
            else:
                # 满足 DOF 在 1 到 6 之间，且驱动器数量 > 0
                json_path = os.path.join("generated_xml", filename.replace(".xml", ".json"))
                m_score = 0.0
                joint_str = "未知构型"  # 新增：用于存储类似 "R1-P1-S1" 的字符串

                if os.path.exists(json_path):
                    # 获取经济性得分
                    econ_res = evaluate_manufacturability(json_path)
                    m_score = econ_res["M_score"]

                    # ==========================================
                    # 【新增逻辑】：解析 JSON 提取关节构型字符串
                    # ==========================================
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # 重建有向图
                        G = nx.node_link_graph(data, directed=True)
                        # 获取拓扑排序 (从基座到动平台)
                        try:
                            topo_order = list(nx.topological_sort(G))
                        except nx.NetworkXUnfeasible:
                            topo_order = list(G.nodes())

                        # 遍历节点，提取含有 'joint_type' 的节点属性
                        joints = []
                        for node_id in topo_order:
                            j_type = G.nodes[node_id].get('joint_type')
                            if j_type:
                                # 去掉一些冗余的后缀数字(如果需要)，或者直接保留原样(如 R1, P1, S1)
                                joints.append(j_type)

                        # 用破折号连接，形成如 "R1-P1-S1" 的标准机器命名格式
                        joint_str = "-".join(joints)
                    except Exception as e:
                        joint_str = "解析错误"
                else:
                    m_score = 0.0
                # 将 joint_str 也存进去
                valid_robots.append((i + 1, filename, dof, model.nu, m_score, joint_str))
        except Exception as e:
            stats["error"] += 1

    # 清除进度条那一行
    print(" " * 80, end="\r")

    # ==========================================
    # 打印最终的分析报告
    # ==========================================
    print("\n" + "=" * 50)
    print(" 📊 批量并联机构筛选分析报告")
    print("=" * 50)
    print(f"总计检测模型数: {total_files} 个")
    print(f"✅ 完美可用机构: {len(valid_robots)} 个 (1 ≤ DOF ≤ 6，且带电机)")
    print("-" * 50)
    print(f"❌ 缺乏驱动器  : {stats['no_actuator']} 个")
    print(f"❌ 死锁或过约束: {stats['locked']} 个 (DOF=0)")
    print(f"❌ 欠约束状态  : {stats['under_constrained']} 个 (DOF>6)")
    print(f"❌ 模型解析崩溃: {stats['error']} 个")
    print("=" * 50)

    # 打印出筛选成功的幸存者名单，方便你手动去查看
    if valid_robots:
        print("\n🌟 正常可用的机器人列表:")
        # 【修改】解包时加入 idx，并在打印时醒目显示编号
        for idx, name, dof, nu, m_score, joint_str in valid_robots:
            print(
                f"  👉 编号 [{idx:02d}] | {name:^18} | 构型: {joint_str:^12} | DOF: {dof} | 电机数: {nu} | M得分: {m_score:.2f}")
    else:
        print("\n🥺 很遗憾，在这一批生成的模型中没有找到完美的结构。你可以尝试多生成一些！")


if __name__ == "__main__":
    analyze_all_robots()