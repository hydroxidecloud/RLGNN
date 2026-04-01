# src/lti_calculator.py

import numpy as np
from scipy.linalg import null_space
import mujoco

# 互易积 (Reciprocal Product) 算子 Delta
DELTA = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.eye(3), np.zeros((3, 3))]
])


def normalize_screw(screw):
    """归一化螺旋向量 (求欧几里得范数)"""
    norm = np.linalg.norm(screw)
    return screw / norm if norm > 1e-6 else screw


def compute_lti_at_current_state(model, data) -> dict:
    """
    计算当前 MuJoCo 仿真状态下的 LTI (局部传递指数)
    注意：在调用此函数前，应确保外部已经执行了 mj_step 或 mj_kinematics
    """

    # 【新增】安全检查：如果物理引擎计算出界（产生了 NaN 坏点），直接返回 0
    if np.isnan(data.xanchor).any() or np.isnan(data.xaxis).any():
        return {"ITI": 0.0, "OTI": 0.0, "LTI": 0.0}

    # 1. 自动识别哪些关节是主动关节 (Actuated Joints)
    actuated_joint_ids = []
    for i in range(model.nu):
        # actuator_trnid[i][0] 存储了第 i 个驱动器绑定的关节 ID
        actuated_joint_ids.append(model.actuator_trnid[i][0])

    branch_twists = {1: {'active': [], 'passive': []},
                     2: {'active': [], 'passive': []},
                     3: {'active': [], 'passive': []}}

    # 2. 遍历提取当前空间状态下的螺旋矩阵，并区分主动/被动
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = model.jnt_type[i]

        if jnt_type == 0 or not jnt_name:
            continue

        b_idx = None
        if "_B1_" in jnt_name:
            b_idx = 1
        elif "_B2_" in jnt_name:
            b_idx = 2
        elif "_B3_" in jnt_name:
            b_idx = 3
        else:
            continue

        r = data.xanchor[i]
        twists = []

        if jnt_type == 3:  # R 关节
            s = data.xaxis[i]
            twists.append(np.concatenate([s, np.cross(r, s)]).reshape(6, 1))
        elif jnt_type == 2:  # P 关节
            s = data.xaxis[i]
            twists.append(np.concatenate([np.zeros(3), s]).reshape(6, 1))
        elif jnt_type == 1:  # S 关节
            axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            for s in axes:
                twists.append(np.concatenate([s, np.cross(r, s)]).reshape(6, 1))

        # 分类存储
        for tw in twists:
            if i in actuated_joint_ids:
                branch_twists[b_idx]['active'].append(tw)
            else:
                branch_twists[b_idx]['passive'].append(tw)

    # 3. 计算各个支链的传递指数 (ITI) 和输出力矩
    iti_list = []
    TWS_list = []

    for b_idx in range(1, 4):
        branch = branch_twists[b_idx]
        if not branch['active'] or not branch['passive']:
            continue

        # 主动螺旋 (ITS - Input Twist Screw)
        ITS = branch['active'][0]

        # 提取该支链所有的被动螺旋
        J_passive = np.hstack(branch['passive'])

        # 传递力螺旋空间 (TWS_space)
        TWS_space = null_space(J_passive.T @ DELTA, rcond=1e-4)

        if TWS_space.size == 0:
            continue  # 如果被动关节已经满秩，无法传递有效力

        # ==========================================
        # 【修改 1】：遍历所有的传递力螺旋，防止遗漏自由度
        # ==========================================
        best_iti = 0.0  # 用于记录该支链最佳的传力效率

        for col in range(TWS_space.shape[1]):
            TWS = TWS_space[:, col].reshape(6, 1)
            TWS_list.append(TWS)  # 收集所有的 TWS，供后续 OTI 完美计算

            # ==========================================
            # 【修改 2】：使用 np.squeeze() 剥离矩阵外壳，解决报错
            # ==========================================
            # 矩阵相乘结果为 [[值]]，squeeze 后变为纯数值，再转 float 就绝对安全了
            reciprocal_product = abs(float(np.squeeze(TWS.T @ DELTA @ ITS)))

            norm_tws = np.linalg.norm(TWS)
            norm_its = np.linalg.norm(ITS)

            if norm_tws > 1e-6 and norm_its > 1e-6:
                iti = reciprocal_product / (norm_tws * norm_its)
            else:
                iti = 0.0

            # 如果有多条力螺旋，我们提取传递效率最高的值作为该支链的评估标准
            best_iti = max(best_iti, iti)

        # 归一化限制在 [0, 1] 之间，并存入支链 ITI 列表
        iti_list.append(min(1.0, best_iti))

    # ==========================================
    # 4. 计算输出传递指数 (OTI) (已修复数学悖论)
    # ==========================================
    # 第一步：收集所有支链的【纯约束力(CWS)】
    # 纯约束力是对支链上*所有关节*（主动+被动）都互易的力
    CWS_list = []
    for b_idx in range(1, 4):
        branch = branch_twists[b_idx]
        if not branch['active'] or not branch['passive']:
            continue
        # 将主动和被动关节全部拼接
        J_all = np.hstack(branch['active'] + branch['passive'])
        # 求纯约束力螺旋
        CWS_space = null_space(J_all.T @ DELTA, rcond=1e-4)
        if CWS_space.size > 0:
            CWS_list.append(CWS_space)

    # 第二步：计算平台允许的运动空间 (OTS)
    if CWS_list:
        CWS_total = np.hstack(CWS_list)
        # 平台的合法运动，必须与所有的纯约束力不发生干涉 (求零空间)
        OTS_space = null_space(CWS_total.T @ DELTA, rcond=1e-4)
    else:
        # 如果没有任何约束，平台是完全自由的 (6自由度)
        OTS_space = np.eye(6)

    # 第三步：计算 OTI (传递推力 TWS 在 平台运动 OTS 上的做功效率)
    oti_list = []
    if OTS_space.size > 0 and TWS_list:
        for TWS in TWS_list:
            max_oti = 0.0
            # 针对该推力，在平台所有可行的运动维度中，找一个最能受力的方向
            for i in range(OTS_space.shape[1]):
                OTS = OTS_space[:, i].reshape(6, 1)

                # 这次的互易积绝对不会永远为0了！
                rp = abs(float(np.squeeze(TWS.T @ DELTA @ OTS)))
                norm_tws = np.linalg.norm(TWS)
                norm_ots = np.linalg.norm(OTS)

                if norm_tws > 1e-6 and norm_ots > 1e-6:
                    oti = rp / (norm_tws * norm_ots)
                    max_oti = max(max_oti, oti)

            oti_list.append(min(1.0, max_oti))
    else:
        # 如果平台被死死卡住(DOF=0)，那么没有任何推力能让它动，OTI 为 0
        oti_list = [0.0]

    # ==========================================
    # 5. 汇总 LTI
    # ==========================================
    # LTI 通常取 ITI 和 OTI 中的短板 (木桶效应)
    final_iti = np.mean(iti_list) if iti_list else 0.0
    final_oti = np.mean(oti_list) if oti_list else 0.0
    final_lti = min(final_iti, final_oti)

    return {
        "ITI": final_iti,
        "OTI": final_oti,
        "LTI": final_lti
    }