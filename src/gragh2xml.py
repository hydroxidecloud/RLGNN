# src/gragh2xml.py

import networkx as nx
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import json
import math

# ==========================================
# 颜色配置表 (RGBA: 显眼且半透明)
# ==========================================
COLOR_LINK = "0.7 0.7 0.7 0.4"
COLOR_R = "0.9 0.2 0.2 0.7"
COLOR_U = "0.2 0.9 0.2 0.7"
COLOR_S = "0.8 0.2 0.8 0.7"
COLOR_P1 = "0.9 0.8 0.2 0.7"
COLOR_PLATFORM = "0.2 0.6 0.9 0.8"


def add_cylinder_geom(parent, axis: str, color: str, size: str = "0.03 0.06"):
    if axis == 'x':
        ET.SubElement(parent, 'geom', type="cylinder", size=size, rgba=color, euler="0 1.5708 0")
    elif axis == 'y':
        ET.SubElement(parent, 'geom', type="cylinder", size=size, rgba=color, euler="1.5708 0 0")
    elif axis == 'z':
        ET.SubElement(parent, 'geom', type="cylinder", size=size, rgba=color)


def load_graph_from_json(filepath: str) -> nx.DiGraph:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True)


def graph_to_mujoco_xml(G: nx.DiGraph) -> str:
    """将 NetworkX 图对象直接转换为 MuJoCo 的 XML 字符串"""
    mujoco = ET.Element('mujoco')
    ET.SubElement(mujoco, 'compiler', angle='radian')
    # 0 0 0 代表处于失重状态（太空环境）
    ET.SubElement(mujoco, 'option', gravity="0 0 0")

    # ==========================================
    # 【新增】：设置全局默认属性，关闭所有 geom 的物理碰撞 (幽灵模式)
    # ==========================================
    default_block = ET.SubElement(mujoco, 'default')
    ET.SubElement(default_block, 'geom', contype="0", conaffinity="0")

    worldbody = ET.SubElement(mujoco, 'worldbody')
    ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")

    # ==========================================
    # 【修改】：给地面单独重新开启碰撞，防止机器人掉进虚空
    # ==========================================
    ET.SubElement(worldbody, 'geom', type="plane", size="2 2 0.1", rgba=".9 .9 .9 1", contype="1", conaffinity="1")

    actuator_block = ET.SubElement(mujoco, 'actuator')
    equality_block = ET.SubElement(mujoco, 'equality')

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        raise ValueError("图中存在环！目前只支持将单链DAG导出为支链。")

    target_actuator_node = None
    j_nodes = [n for n in topo_order if 'joint_type' in G.nodes[n]]

    for n in j_nodes:
        if G.nodes[n].get('joint_type') == 'P1':
            target_actuator_node = n
            break

    if target_actuator_node is None:
        for n in j_nodes:
            j_type = G.nodes[n].get('joint_type')
            if j_type in ['R1', 'R2', 'R3']:
                target_actuator_node = n
                break

    branch_angles = {
        1: "0 0 0",
        2: f"0 0 {2 * math.pi / 3}",
        3: f"0 0 {4 * math.pi / 3}"
    }

    for b_idx in range(1, 4):
        euler_str = branch_angles[b_idx]
        branch_root = ET.SubElement(worldbody, 'body', name=f"Branch_{b_idx}", pos="0 0 0", euler=euler_str)
        current_parent = branch_root

        for node_id in topo_order:
            data = G.nodes[node_id]
            symbol = data.get('symbol')

            if symbol == 'E':
                offset = data.get('offset', 0)
                current_parent = ET.SubElement(current_parent, 'body', name=f"Base_B{b_idx}_{node_id}", pos=f"{offset} 0 0")

            elif symbol == 'L':
                length = data.get('length', 1.0)
                ET.SubElement(current_parent, 'geom', type="cylinder", fromto=f"0 0 0 0 0 {length}", size="0.04", rgba=COLOR_LINK)
                current_parent = ET.SubElement(current_parent, 'body', name=f"Link_B{b_idx}_{node_id}", pos=f"0 0 {length}")

            elif symbol == 'P':
                ET.SubElement(current_parent, 'site', name=f"end_effector_B{b_idx}", pos="0 0 0", size="0.02", rgba="1 0 0 1")

            elif 'joint_type' in data:
                j_type = data.get('joint_type')
                joint_name = f"J_B{b_idx}_{node_id}"

                if j_type in ['R1', 'R2', 'R3', 'P1']:
                    theta = data.get('theta_int', 0.0)
                    if j_type == 'R1':
                        current_parent.set('euler', f"0 0 {theta}")
                        ET.SubElement(current_parent, 'joint', name=joint_name, type="hinge", axis="0 0 1", limited="true", range="-1.5708 1.5708")
                        add_cylinder_geom(current_parent, 'z', COLOR_R)
                    elif j_type == 'R2':
                        current_parent.set('euler', f"0 {theta} 0")
                        ET.SubElement(current_parent, 'joint', name=joint_name, type="hinge", axis="0 1 0", limited="true", range="-1.5708 1.5708")
                        add_cylinder_geom(current_parent, 'y', COLOR_R)
                    elif j_type == 'R3':
                        current_parent.set('euler', f"{theta} 0 0")
                        ET.SubElement(current_parent, 'joint', name=joint_name, type="hinge", axis="1 0 0", limited="true", range="-1.5708 1.5708")
                        add_cylinder_geom(current_parent, 'x', COLOR_R)
                    elif j_type == 'P1':
                        pos_str = current_parent.get('pos', '0 0 0')
                        px, py, pz = map(float, pos_str.split())
                        # current_parent.set('pos', f"{px} {py} {pz + theta}")
                        # 不加入随机的关节初始变量，因为前后link的长度已经能满足其变化。
                        current_parent.set('pos', f"{px} {py} {pz}")
                        # pz 在这里就是上一个 link 的长度。
                        # 我们利用 f-string 动态生成 range 字符串，形式为 "负的长度 0.0"
                        # 保留三位小数，防止浮点数格式过长报错
                        slide_range = f"{-pz:.3f} 0.0"
                        ET.SubElement(current_parent, 'joint', name=joint_name, type="slide", axis="0 0 1",
                                      limited="true", range=slide_range)
                        add_cylinder_geom(current_parent, 'z', COLOR_P1, size="0.04 0.08")

                elif j_type.startswith('U'):
                    theta1 = data.get('theta_int1', 0.0)
                    theta2 = data.get('theta_int2', 0.0)
                    if j_type == 'U1':
                        current_parent.set('euler', f"{theta1} {theta2} 0")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_x", type="hinge", axis="1 0 0", limited="true", range="-0.785 0.785")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_y", type="hinge", axis="0 1 0", limited="true", range="-0.785 0.785")
                        add_cylinder_geom(current_parent, 'x', COLOR_U)
                        add_cylinder_geom(current_parent, 'y', COLOR_U)
                    elif j_type == 'U2':
                        current_parent.set('euler', f"{theta1} 0 {theta2}")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_x", type="hinge", axis="1 0 0", limited="true", range="-0.785 0.785")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_z", type="hinge", axis="0 0 1", limited="true", range="-0.785 0.785")
                        add_cylinder_geom(current_parent, 'x', COLOR_U)
                        add_cylinder_geom(current_parent, 'z', COLOR_U)
                    elif j_type == 'U3':
                        current_parent.set('euler', f"0 {theta1} {theta2}")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_y", type="hinge", axis="0 1 0", limited="true", range="-0.785 0.785")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_z", type="hinge", axis="0 0 1", limited="true", range="-0.785 0.785")
                        add_cylinder_geom(current_parent, 'y', COLOR_U)
                        add_cylinder_geom(current_parent, 'z', COLOR_U)
                    elif j_type == 'U4':
                        current_parent.set('euler', f"{theta2} {theta1} 0")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_y", type="hinge", axis="0 1 0", limited="true", range="-0.785 0.785")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_x", type="hinge", axis="1 0 0", limited="true", range="-0.785 0.785")
                        add_cylinder_geom(current_parent, 'y', COLOR_U)
                        add_cylinder_geom(current_parent, 'x', COLOR_U)
                    elif j_type == 'U5':
                        current_parent.set('euler', f"{theta2} 0 {theta1}")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_z", type="hinge", axis="0 0 1", limited="true", range="-0.785 0.785")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_x", type="hinge", axis="1 0 0", limited="true", range="-0.785 0.785")
                        add_cylinder_geom(current_parent, 'z', COLOR_U)
                        add_cylinder_geom(current_parent, 'x', COLOR_U)
                    elif j_type == 'U6':
                        current_parent.set('euler', f"0 {theta2} {theta1}")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_z", type="hinge", axis="0 0 1", limited="true", range="-0.785 0.785")
                        ET.SubElement(current_parent, 'joint', name=f"{joint_name}_y", type="hinge", axis="0 1 0", limited="true", range="-0.785 0.785")
                        add_cylinder_geom(current_parent, 'z', COLOR_U)
                        add_cylinder_geom(current_parent, 'y', COLOR_U)

                elif j_type == 'S1':
                    theta1 = data.get('theta_int1', 0.0)
                    theta2 = data.get('theta_int2', 0.0)
                    theta3 = data.get('theta_int3', 0.0)
                    current_parent.set('euler', f"{theta1} {theta2} {theta3}")
                    ET.SubElement(current_parent, 'joint', name=joint_name, type="ball", limited="true", range="0 0.5")
                    ET.SubElement(current_parent, 'geom', type="sphere", size="0.05", rgba=COLOR_S)

                if node_id == target_actuator_node:
                    ET.SubElement(
                        actuator_block,
                        'motor',  # 1. 改为力矩电机
                        name=f"Motor_{joint_name}",
                        joint=joint_name,
                        gear="100",  # 2. 加入减速比放大器(齿轮比)，提供足够的力量
                        ctrllimited="true", ctrlrange="-10.0 10.0"  # 3. 放大控制域
                    )

    platform_radius = 0.2
    platform_z_guess = 1.5

    platform = ET.SubElement(worldbody, 'body', name="Central_Moving_Platform", pos=f"0 0 {platform_z_guess}")
    ET.SubElement(platform, 'freejoint', name="platform_freejoint")
    ET.SubElement(platform, 'geom', type="cylinder", size=f"{platform_radius} 0.025", rgba=COLOR_PLATFORM, contype="1", conaffinity="1")

    for b_idx in range(1, 4):
        angle = (b_idx - 1) * (2 * math.pi / 3)
        px = platform_radius * math.cos(angle)
        py = platform_radius * math.sin(angle)
        ET.SubElement(platform, 'site', name=f"platform_attach_B{b_idx}", pos=f"{px} {py} 0", size="0.02", rgba="0 1 0 1")
        ET.SubElement(equality_block, 'connect', site1=f"end_effector_B{b_idx}", site2=f"platform_attach_B{b_idx}")

    rough_string = ET.tostring(mujoco, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return '\n'.join([line for line in reparsed.toprettyxml(indent="  ").split('\n') if line.strip()])


def save_mujoco_xml(xml_content: str, filepath: str):
    """保存 XML 文件到指定的路径"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xml_content)