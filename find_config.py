import os
import json
import networkx as nx


def search_robot_configuration(target_config, json_dir="generated_robots", xml_dir="generated_xml"):
    """
    遍历所有的 JSON 图纸文件，查找特定构型的机器人，并映射出用于仿真的序号。
    """
    target_config = target_config.strip().upper()
    print(f"\n🔍 正在全网搜寻包含构型特征 '{target_config}' 的机器人图纸...")

    if not os.path.exists(json_dir):
        print(f"❌ 找不到 JSON 目录 '{json_dir}'！请确认你的文件生成路径。")
        return

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    # 提前获取 xml 文件的排序列表，用于精确计算“仿真序号”
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')]) if os.path.exists(xml_dir) else []

    found_matches = []

    for filename in json_files:
        json_path = os.path.join(json_dir, filename)
        xml_name = filename.replace('.json', '.xml')
        xml_path = os.path.join(xml_dir, xml_name)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 重建有向图
            G = nx.node_link_graph(data, directed=True)

            # 拓扑排序 (从基座到动平台)
            try:
                topo_order = list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                topo_order = list(G.nodes())

            # 提取关节序列
            joints = []
            for node_id in topo_order:
                j_type = G.nodes[node_id].get('joint_type')
                if j_type:
                    joints.append(j_type)

            # 拼接成字符串
            joint_str = "-".join(joints)

            # 模糊匹配
            if target_config in joint_str:
                xml_exists = os.path.exists(xml_path)
                # 计算它在 test_parallel 交互界面中的真实序号
                seq_idx = xml_files.index(xml_name) + 1 if xml_exists and xml_name in xml_files else 0

                found_matches.append((seq_idx, xml_name, joint_str, xml_exists))

        except Exception as e:
            continue

    # ==========================================
    # 打印搜索报告
    # ==========================================
    if found_matches:
        print(f"\n🎉 搜索完毕！共找到 {len(found_matches)} 个匹配的模型：")
        print("=" * 80)
        for seq_idx, xml_name, j_str, xml_exists in found_matches:
            status = "✅ 可仿真" if xml_exists else "⚠️ 无XML"
            # 这里的 seq_idx 就是你可以直接在 MPPI 终端里敲的数字！
            seq_str = f"[{seq_idx:02d}]" if seq_idx > 0 else "[--]"
            print(f"  👉 序号: {seq_str} | 文件: {xml_name:^18} | 构型: {j_str:^15} | 状态: {status}")
        print("=" * 80)
        print("💡 提示: 记住前面的【序号】，直接去 test_parallel_MPPI.py 终端里输入它即可启动智能控制！")
    else:
        print(f"\n🥺 搜索完毕。没有找到包含 '{target_config}' 的机器人。")


if __name__ == "__main__":
    print("🤖 欢迎使用并联机构构型雷达！")
    while True:
        target = input("\n👉 请输入你想查找的构型 (如 R2-P1-S1，输入 q 退出): ")
        if target.lower() == 'q':
            break
        if target.strip() == "":
            continue

        search_robot_configuration(target)