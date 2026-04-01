# src/calc_screw_dof.py

import os
import numpy as np
from scipy.linalg import null_space
import mujoco

# ==========================================
# 螺旋理论 (Screw Theory) 核心算子
# ==========================================
# 定义互易积 (Reciprocal Product) 算子 Delta
# Twist = [w, v]^T, Wrench = [f, m]^T
# 互易条件: Twist^T * Delta * Wrench = 0
DELTA = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.eye(3), np.zeros((3, 3))]
])


def compute_screw_theory_dof(xml_path: str, run_sim: bool = True) -> int:
    """
    基于螺旋理论和 MuJoCo 运动学树计算并联机构的 DOF
    :param xml_path: XML 模型的路径
    :param run_sim: 是否在计算完成后自动启动物理仿真演示
    :return: 计算得出的机构自由度 (int)
    """
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return -1

    # 进行一次纯运动学正向计算，确保所有几何体、坐标系、关节在第 0 帧就位
    mujoco.mj_kinematics(model, data)

    # print(f"\n" + "=" * 60)
    # print(f" ⚙️ 螺旋理论运动学分析: {os.path.basename(xml_path)}")
    # print("=" * 60)

    # 用字典收集三条支链的运动螺旋 (Twists)
    branch_twists = {1: [], 2: [], 3: []}

    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = model.jnt_type[i]

        # 跳过动平台的 freejoint (类型 0)
        if jnt_type == 0 or not jnt_name:
            continue

        # 判断该关节属于哪条支链
        b_idx = None
        if "_B1_" in jnt_name:
            b_idx = 1
        elif "_B2_" in jnt_name:
            b_idx = 2
        elif "_B3_" in jnt_name:
            b_idx = 3
        else:
            continue

        # 获取关节在世界坐标系下的绝对位置 r 和 绝对方向 s
        r = data.xanchor[i]

        if jnt_type == 3:  # Hinge (转动关节 R)
            s = data.xaxis[i]
            v = np.cross(r, s)
            twist = np.concatenate([s, v]).reshape(6, 1)
            branch_twists[b_idx].append(twist)

        elif jnt_type == 2:  # Slide (移动关节 P)
            s = data.xaxis[i]
            w = np.zeros(3)
            twist = np.concatenate([w, s]).reshape(6, 1)
            branch_twists[b_idx].append(twist)

        elif jnt_type == 1:  # Ball (球面关节 S)
            axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            for s in axes:
                v = np.cross(r, s)
                twist = np.concatenate([s, v]).reshape(6, 1)
                branch_twists[b_idx].append(twist)

    # 汇总各支链的约束力矩 (Wrenches)
    W_total_list = []

    for b_idx in range(1, 4):
        twists = branch_twists[b_idx]
        if not twists:
            continue

        J_i = np.hstack(twists)
        q_i = J_i.shape[1]

        operator_matrix = J_i.T @ DELTA
        W_i = null_space(operator_matrix, rcond=1e-4)
        num_constraints = W_i.shape[1] if W_i.size > 0 else 0

        # print(f"🔹 支链 {b_idx}: 包含 {q_i} 个独立运动螺旋 (DOF={q_i})")
        # print(f"   └─ 对平台施加了 {num_constraints} 个约束力矩 (Constraints)")

        if num_constraints > 0:
            W_total_list.append(W_i)

    final_dof = 0

    if not W_total_list:
        # print("\n⚠️ 未检测到任何约束力矩，动平台似乎处于完全无约束的自由状态 (DOF=6)！")
        final_dof = 6
    else:
        # 将三条支链的约束力矩矩阵拼接在一起 (6 x 总约束数)
        W_total = np.hstack(W_total_list)
        total_constraints = W_total.shape[1]

        # 二次零空间：计算动平台的输出运动螺旋 O$ (Output Twist Space)
        final_operator = W_total.T @ DELTA
        Twist_out = null_space(final_operator, rcond=1e-4)

        # 动平台的最终 DOF 即为输出运动螺旋矩阵的列秩 (Rank)
        final_dof = Twist_out.shape[1] if Twist_out.size > 0 else 0

        # print("-" * 60)
        # print(f"🌐 平台全局力矩矩阵合并: 总计存在 {total_constraints} 个理论约束")
        # print(f"🔥 基于螺旋理论计算得出，该并联机构动平台的实际 DOF = 【 {final_dof} 】")
        # print("-" * 60)

        # if final_dof == 0:
        #     print("💡 解析：这是一个过约束/死锁结构，动平台在理论上无法发生运动。")
        # elif final_dof == 6:
        #     print("💡 解析：这相当于一个六自由度并联机构 (如 Stewart 平台)。")
        # elif final_dof == 3:
        #     print("💡 解析：这是一个典型的三自由度并联机构 (如 Delta, 3-RPS 等)。")

    # 根据开关决定是否播放动画
    if run_sim:
        print("\n▶️ 理论计算结束，自动进入物理仿真演示环节...")
        run_simulation(model, data)

    return final_dof


def run_simulation(model, data):
    import time
    try:
        from mujoco import viewer
        print("🚀 启动 MuJoCo 图形界面，按窗口右上角关闭...")
        num_actuators = model.nu
        duration = 100.0

        with viewer.launch_passive(model, data) as v:
            while v.is_running() and data.time < duration:
                step_start = time.time()

                # 同步正弦波驱动
                if num_actuators > 0:
                    import math
                    for i in range(num_actuators):
                        data.ctrl[i] = 0.2 * math.sin(math.pi * data.time)

                mujoco.mj_step(model, data)
                v.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        print(f"⚠️ 无法启动图形界面: {e}")


def select_and_calculate():
    """
    提供给终端的交互式菜单
    """
    xml_dir = "generated_xml"

    if not os.path.exists(xml_dir):
        print(f"❌ 找不到目录 '{xml_dir}'！")
        return

    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])

    if not xml_files:
        print(f"⚠️ 目录 '{xml_dir}' 中没有任何 XML 文件。")
        return

    print(f"\n📂 请选择需要计算自由度的模型：")
    print("=" * 50)
    for i, filename in enumerate(xml_files):
        print(f"  [{i + 1}] {filename}")
    print("=" * 50)

    while True:
        try:
            choice = input(f"\n👉 请输入编号 (1-{len(xml_files)})，或输入 'q' 退出: ")

            if choice.lower() == 'q':
                break

            idx = int(choice) - 1
            if 0 <= idx < len(xml_files):
                selected_xml_path = os.path.join(xml_dir, xml_files[idx])
                # 这里默认开启 run_sim=True
                compute_screw_theory_dof(selected_xml_path, run_sim=True)
                break
            else:
                print("❌ 编号超出范围，请重新输入！")
        except ValueError:
            print("❌ 无效的输入！请输入数字编号。")


if __name__ == "__main__":
    # 允许单独运行此文件时触发交互菜单
    select_and_calculate()