# src/simulator.py
import os
import math
import mujoco
import numpy as np
from tinydb import TinyDB, Query


# ==========================================
# 📊 带有 MPC 大脑的极速无头物理仿真器
# ==========================================
def evaluate_robot_headless(xml_path):
    """
    基于 MuJoCo 和随机打靶法 (Random Shooting MPC) 的智能控制器。
    它会在脑海里推演未来，自主寻找最优发力序列，以最大化中央平台的 Z 轴行程。
    """
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception:
        # 如果随机生成的机器人物理拓扑极其奇葩导致 MuJoCo 崩溃，直接淘汰给 0 分
        return 0.0

    num_motors = model.nu
    if num_motors == 0:
        return 0.0

    try:
        platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Central_Moving_Platform")
    except:
        return 0.0

    # ==========================================
    # 🧠 MPC 核心超参数配置
    # ==========================================
    # 注意：MPC 计算量极大，因此我们把现实中跑的总时长缩短，否则你的电脑会算到冒烟
    sim_steps = 500  # 现实世界总共走多少步 (比如 500步 * 0.002s = 1秒)
    horizon = 10  # 每次在脑海里往未来预测 10 步
    num_samples = 32  # 每次在脑海里生成 32 种不同的发力可能
    ctrl_range = 10.0  # 对应你在 gragh2xml 里的力矩上限 (-10.0 到 10.0)

    max_z = float('-inf')
    min_z = float('inf')

    # 预先分配好一块用来做“脑内推演”的内存，极大加速运算
    d_predict = mujoco.MjData(model)

    # 记录初始高度，方便计算偏移量
    mujoco.mj_forward(model, data)
    init_z = data.xpos[platform_id][2]

    # ==========================================
    # 🌍 主物理循环 (现实时间的流逝)
    # ==========================================
    for step in range(sim_steps):
        best_first_action = np.zeros(num_motors)
        best_score = -float('inf')

        # 一次性生成所有样本的随机控制序列 [样本数, 预测步数, 电机数]
        action_sequences = np.random.uniform(-ctrl_range, ctrl_range, size=(num_samples, horizon, num_motors))

        # 🔮 开始脑内推演
        for i in range(num_samples):
            # 1. 将预测世界重置为当前的现实状态 (极速状态拷贝)
            # 利用 NumPy 的切片赋值，直接覆盖底层 C 内存，完美替代废弃的 mj_copyData
            d_predict.time = data.time
            d_predict.qpos[:] = data.qpos[:]
            d_predict.qvel[:] = data.qvel[:]

            # 如果你的机器人包含了肌肉模型或有内部状态的复杂马达，还需要拷贝 act 状态
            if model.na > 0:
                d_predict.act[:] = data.act[:]

            # 如果希望求解器在推演第一步时更稳定（继承上一步的接触力缓存），可以选加这一行：
            d_predict.qacc_warmstart[:] = data.qacc_warmstart[:]

            simulated_score = 0.0
            is_broken = False

            # 2. 在脑海里跑完这段未来
            for h in range(horizon):
                d_predict.ctrl[:] = action_sequences[i, h]
                mujoco.mj_step(model, d_predict)

                # 获取推演中的平台高度
                current_z = d_predict.xpos[platform_id][2]

                # ⭐ 奖励函数 (Reward Function)
                # 我们的目标是最大化位移（向上或向下都可以，只要偏离初始位置越远越好）
                simulated_score += abs(current_z - init_z)

                # ☠️ 惩罚机制：继承你原来的“10000N 淘汰机制”
                if d_predict.efc_force.size > 0:
                    max_force = max(abs(d_predict.efc_force))
                    if max_force > 10000.0:
                        simulated_score -= 1000.0  # 爆炸了，给极低分，这套动作作废
                        is_broken = True
                        break

            # 3. 如果这个平行宇宙的结局最好，记住它的第一步怎么走的！
            if simulated_score > best_score:
                best_score = simulated_score
                best_first_action = action_sequences[i, 0]

        # ==========================================
        # ⚡ 现实执行：应用想出来的最完美的第一步
        # ==========================================
        data.ctrl[:] = best_first_action
        mujoco.mj_step(model, data)

        # 记录现实世界中的真实极限行程
        real_z = data.xpos[platform_id][2]
        max_z = max(max_z, real_z)
        min_z = min(min_z, real_z)

        # 现实世界的断裂判定
        if data.efc_force.size > 0:
            current_max_force = max(abs(data.efc_force))
            if current_max_force > 10000.0:
                break  # 骨折了，立刻终止这台机器人的生命

    # 计算最终性能值 (真实的物理 Stroke)
    if max_z == float('-inf') or min_z == float('inf'):
        return 0.0

    return max_z - min_z


# ==========================================
# 💾 TinyDB 仿真数据入库模块 (保持完全不变)
# ==========================================
def save_simulation_to_tinydb(xml_path, score, db_path='data/simulation_table.json'):
    # ... (原有代码保持完全不变)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = TinyDB(db_path)
    Robot = Query()
    design_id = os.path.basename(xml_path).replace('.xml', '.json')
    result = db.search(Robot.design_id == design_id)

    if not result:
        db.insert({
            'design_id': design_id,
            'eval_count': 1,
            'sum_v': score,
            'mean_v': score,
            'max_mean_v': score
        })
    else:
        doc = result[0]
        new_count = doc['eval_count'] + 1
        new_sum = doc['sum_v'] + score
        new_mean = new_sum / new_count
        new_max = max(new_mean, doc['max_mean_v'])

        db.update({
            'eval_count': new_count,
            'sum_v': new_sum,
            'mean_v': new_mean,
            'max_mean_v': new_max
        }, Robot.design_id == design_id)
    db.close()