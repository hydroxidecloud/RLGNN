import os
import math
import mujoco
from src.calc_screw_dof import compute_screw_theory_dof


def diagnostic_simulation(xml_path):
    """
    专门用于体检排错的透明化物理仿真，会返回准确的死因
    """
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        return False, 0.0, f"MuJoCo底层加载XML崩溃: {e}"

    platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Central_Moving_Platform")
    if platform_id == -1:
        return False, 0.0, "找不到名为 'Central_Moving_Platform' 的输出平台"

    # === 找到这部分代码 ===
    sine_period = 10.0

    # 🌟 新增：初始调整时长（从 1.0 改为 2.0 或 3.0 秒）
    init_duration = 3.0

    duration = init_duration + sine_period
    max_z = float('-inf')
    min_z = float('inf')

    # 清空可能存在的历史警告
    data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 0
    data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number = 0

    while data.time < duration:
        # 🌟 修改：使用 init_duration 判断
        if data.time < init_duration:
            data.ctrl[:] = 0.0
        else:
            active_time = data.time - init_duration
            torque = 10.0 * math.sin((2 * math.pi / sine_period) * active_time)
            data.ctrl[:] = torque

            current_z = data.xpos[platform_id][2]
            if current_z > max_z: max_z = current_z
            if current_z < min_z: min_z = current_z

        # 步进物理引擎
        try:
            mujoco.mj_step(model, data)
        except Exception as e:
            return False, 0.0, f"💥 在 {data.time:.3f}秒 时代码崩溃: {e}"

        # ... (中间的侦测 1 保持不变) ...

        # ----------------------------------------------------
        # 🕵️‍♂️ 深度侦测 2：内部极限应力 (只测初始化结束以后)
        # ----------------------------------------------------
        # 🌟 修改：使用 init_duration 判断
        if data.time >= init_duration and data.nefc > 0:
            current_max_force = max(abs(data.efc_force))
            if current_max_force > 10000.0:
                return False, 0.0, f"💥 在 {data.time:.3f}秒 内部约束力爆表！当前最大受力: {current_max_force:.1f} N (安全阈值 10000.0 N)。机构陷入内耗对抗。"

    # ----------------------------------------------------
    # 🕵️‍♂️ 深度侦测 3：装死（完全不连动）
    # ----------------------------------------------------
    stroke = max_z - min_z
    if stroke < 1e-4:  # 行程小于 0.1 毫米
        return False, 0.0, f"💤 机构一动不动！虽然受力安全，但平台总行程仅为 {stroke:.6f} m。可能是驱动器未正确传递动力到平台。"

    return True, stroke, "运行极其顺畅"


def test_single_xml(xml_path: str):
    if not os.path.exists(xml_path):
        print(f"❌ 找不到文件: {xml_path}")
        return False

    print(f"\n" + "=" * 60)
    print(f" 🔍 开始对机器人进行全流程深度体检: {os.path.basename(xml_path)}")
    print("=" * 60)

    # 1. 基础检测
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        if model.nu == 0:
            print("  ❌ [淘汰] 预检失败：没有驱动器 (nu = 0)。")
            return False
        print(f"  ✅ [通过] 基础检测：包含 {model.nu} 个驱动器。")
    except Exception as e:
        print(f"  ❌ [淘汰] 预检失败：XML 损坏 -> {e}")
        return False

    # 2. 运动学 DOF
    dof = compute_screw_theory_dof(xml_path, run_sim=False)
    if dof <= 0:
        print(f"  ❌ [淘汰] 运动学失败：死锁或过约束 (DOF = {dof})。")
        return False
    elif dof > 6:
        print(f"  ❌ [淘汰] 运动学失败：欠约束或散架 (DOF = {dof})。")
        return False
    print(f"  ✅ [通过] 运动学检测：自由度正常 (DOF = {dof})。")

    # 3. 诊断级物理仿真
    print("  ⏳ 正在启动 X光级透明仿真引擎...")
    success, score, reason = diagnostic_simulation(xml_path)

    if success:
        print(f"  ✅ [通过] 仿真完成：机器人在安全受力下运行正常！")
        print(f"  🏆 最终评估成绩 (总行程): {score:.4f} m")
        return True
    else:
        print(f"\n  🚨 >>> 确切死因查明 <<< 🚨")
        print(f"  {reason}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_xml = sys.argv[1]
    else:
        print("💡 提示：你可以直接拖拽 XML 文件到终端窗口中")
        target_xml = input("👉 请输入要测试的 XML 文件路径: ").strip()
        target_xml = target_xml.strip('"').strip("'")

    success = test_single_xml(target_xml)

    print("\n" + "*" * 60)
    if success:
        print(" 🎉 最终结论：该机器人【完美通过】测试！")
    else:
        print(" 🗑️ 最终结论：该机器人【未能通过】体检。")
    print("*" * 60 + "\n")