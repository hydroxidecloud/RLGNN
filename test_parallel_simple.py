# test_parallel.py
import time
import math
import json
import networkx as nx
import os
import numpy as np
from src.lti_calculator import compute_lti_at_current_state
from src.calc_screw_dof import compute_screw_theory_dof
import mujoco
from tinydb import TinyDB, Query

# ==========================================
# 💾 TinyDB 仿真数据入库模块
# ==========================================
def save_simulation_to_tinydb(xml_path, score, db_path='simulation_table.json'):
    db = TinyDB(db_path)
    Robot = Query()

    # 【极其重要】：把 .xml 后缀换成 .json，确保和 GNN 脚本里的 design_id 完美匹配！
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
        print(f"💾 [TinyDB] 发现新图纸的仿真数据！已创建记录 [{design_id}], 真实性能 V: {score:.4f}")
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

        print(f"💾 [TinyDB] [{design_id}] 第 {new_count} 次仿真记录已更新！当前均值 V: {new_mean:.4f}")

# ==========================================
# 1. 定义 MPPI 控制器
# ==========================================
class SimpleMPPI:
    def __init__(self, model, nu, horizon=1, num_samples=2000, noise_sigma=2.0, lambda_=1.0, sim_steps_per_ctrl=5):
        """
        :param horizon: 预测未来多少个控制步
        :param num_samples: 每次采样多少条随机轨迹
        :param noise_sigma: 动作空间的探索噪声方差
        :param sim_steps_per_ctrl: 每一个控制步对应多少个 MuJoCo 物理步 (拉长预测时间)
        """
        self.model = model
        self.nu = nu
        self.horizon = horizon
        self.K = num_samples
        self.sigma = noise_sigma
        self.lambda_ = lambda_
        self.sim_steps_per_ctrl = sim_steps_per_ctrl

        # 预分配一个独立的 MjData 用于后台“推演”轨迹，互不干扰
        self.d_sim = mujoco.MjData(model)

        # 初始的控制序列均值矩阵 (Horizon x 动作维度)
        self.U = np.zeros((self.horizon, self.nu))

        # 获取中央动平台的 ID，用于提取它的位置和姿态
        self.platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Central_Moving_Platform")

    def compute_control(self, current_data, target_z):
        if self.nu == 0: return []

        costs = np.zeros(self.K)
        # 生成 K 条轨迹的随机高斯噪声
        single_dim_noises = np.random.normal(0, self.sigma, (self.K, self.horizon, 1))
        # 然后把这 1 个数值无脑复制（广播）给所有的电机，保证它们同呼吸共命运
        noises = np.repeat(single_dim_noises, self.nu, axis=2)

        for k in range(self.K):
            # 稳健版：手动将当前真实的物理状态同步到推演引擎中
            self.d_sim.time = current_data.time
            self.d_sim.qpos[:] = current_data.qpos[:]
            self.d_sim.qvel[:] = current_data.qvel[:]
            if self.model.na > 0:  # 如果有高级激活状态（如肌肉），也同步过来
                self.d_sim.act[:] = current_data.act[:]

            # 执行一次正向运动学，刷新 d_sim 里的所有 xpos, xmat 空间坐标
            mujoco.mj_forward(self.model, self.d_sim)

            cost = 0.0
            for t in range(self.horizon):
                # 动作 = 当前最优规划均值 + 随机噪声探索
                ctrl = self.U[t] + noises[k, t]
                ctrl = np.clip(ctrl, -10.0, 10.0)  # 电机限幅

                self.d_sim.ctrl[:] = ctrl

                # 步进物理引擎
                for _ in range(self.sim_steps_per_ctrl):
                    mujoco.mj_step(self.model, self.d_sim)

                # ==========================================
                # 核心：定义目标价值函数 (Cost Function)
                # ==========================================
                # 1. 高度追踪 (要求 Z 坐标贴近 target_z)
                current_z = self.d_sim.xpos[self.platform_id][2]
                z_error = (current_z - target_z) ** 2
                cost += z_error * 2000.0  # 极大的权重，逼迫平台上下走

                # 2. 姿态惩罚 (要求平台保持水平)
                # d_sim.xmat 是 3x3 的旋转矩阵 (压平为9维数组)。索引 8 代表 Z轴方向的余弦值。
                # 如果平台完全水平，Z轴朝上，xmat[8] 应该是 1.0。
                # (修复后的新代码)
                # 先取出对应 platform_id 的旋转矩阵，展平后取第 8 个元素 (即 ZZ 轴)
                tilt_error = 1.0 - self.d_sim.xmat[self.platform_id].reshape(-1)[8]
                cost += tilt_error * 500.0  # 惩罚倾斜

                # 3. 能量惩罚 (让动作尽可能平滑，不要猛踩油门)
                cost += self.lambda_ * np.sum(ctrl ** 2) * 0.1

            costs[k] = cost

        # ==========================================
        # 根据推演的 Cost，计算每条轨迹的权重
        # ==========================================
        beta = np.min(costs)  # 减去最小值防止 exp 溢出
        weights = np.exp(-1.0 / self.lambda_ * (costs - beta))
        weights /= (np.sum(weights) + 1e-8)

        # ==========================================
        # 提取最优轨迹，更新控制序列均值
        # ==========================================
        for t in range(self.horizon):
            self.U[t] += np.sum(weights[:, None] * noises[:, t, :], axis=0)

        # 取出第一步作为当前的实际执行动作
        action = self.U[0].copy()

        # 时间序列前移一位，为下一次控制做准备 (Warm Start)
        self.U[:-1] = self.U[1:]
        self.U[-1] = np.zeros(self.nu)

        return np.clip(action, -10.0, 10.0)


# ==========================================
# 2. 原始的打印和运行逻辑
# ==========================================
def print_graph_info(json_filepath: str):
    # (保持原样，省略了中间代码以突出重点...)
    if not os.path.exists(json_filepath): return
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=True)
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo_order = list(G.nodes())
    print(f"✅ 成功加载原始图数据: {json_filepath}")


def test_parallel_robot(xml_path: str):
    # 1. 载入模型
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    num_actuators = model.nu
    print(f"\n✅ 模型加载成功！检测到 {num_actuators} 个驱动器。")

    # 先做一次正向运动学解算，获取平台最初始、自然的 Z 轴高度
    mujoco.mj_kinematics(model, data)
    platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Central_Moving_Platform")
    base_z = data.xpos[platform_id][2]
    print(f"🎯 平台初始基准高度: {base_z:.3f} m")

    # 初始化 MPPI 控制器
    mppi_controller = SimpleMPPI(model, num_actuators)

    # 设定正弦波的周期为 10 秒 (时间长一点)
    sine_period = 5.0
    # 总运行时间 = 1秒初始化 + 1个完整正弦周期
    duration = 1.0 + sine_period * 10

    # 准备记录 Z 轴极值的变量
    max_z = float('-inf')
    min_z = float('inf')

    try:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as v:
            print("🚀 图形界面运行中...")
            print("🧠 AI 正使用 MPPI 算法控制平台做纯净的上下平移...")

            while v.is_running() and data.time < duration:
                step_start = time.time()

                if num_actuators > 0:
                    if data.time < 1.0:
                        # ==========================================
                        # 🛡️ 第 1 阶段：1秒钟的“完全自由掉落/装配期”
                        # ==========================================
                        # 扭矩控制下，指令为 0 代表电机彻底断电，关节完全自由顺从
                        data.ctrl[:] = 0.0

                        # 持续跟踪平台在重力和约束下自由沉降后的最终高度。
                        # 当 time 达到 1.0s 的那一瞬间，这个 base_z 就是机构最自然、最放松的起点！
                        base_z = data.xpos[platform_id][2]
                        # print("data.time < 1.0，初始化进行中")

                    else:
                        # ==========================================
                        # 🌊 第 2 阶段：长周期正弦波控制与极值记录
                        # ==========================================
                        active_time = data.time - 1.0

                        # 正弦波公式: Torque = Amplitude * sin(2 * pi * t / T)
                        amplitude = 10.0  # 依然保持 1000 的峰值扭矩
                        torque = amplitude * math.sin((2 * math.pi / sine_period) * active_time)

                        # 给电机施加当前时刻的正弦扭矩
                        data.ctrl[:] = torque

                        # 实时获取当前平台的 Z 轴高度
                        current_z = data.xpos[platform_id][2]

                        # 记录极值
                        if current_z > max_z:
                            max_z = current_z
                        if current_z < min_z:
                            min_z = current_z
                        # 注：不需要控制 step_counter 了，因为方波每帧都可以直接覆盖输出

                # 步进物理引擎
                mujoco.mj_step(model, data)
                v.sync()

                # 在 mj_step 之后加上这句看看
                # print("最大约束力:", max(abs(data.efc_force)))

                # LTI 计算 (为了不卡顿，频率调低一点)
                if int(data.time / model.opt.timestep) % int(1.0 / model.opt.timestep) == 0:
                    try:
                        transmission_perf = compute_lti_at_current_state(model, data)
                        lti_value = transmission_perf['LTI']
                        status = "✅ 正常" if lti_value > 0.1 else "⚠️ 接近奇异 (Singularity)"
                        print(
                            f"⏱️ 时间: {data.time:.2f}s | ITI: {transmission_perf['ITI']:.3f} | OTI: {transmission_perf['OTI']:.3f} | LTI: {lti_value:.3f} | {status} | 最大约束力: {max(abs(data.efc_force)):.3f} ")

                    except Exception as e:
                        # 捕获报错，但不退出循环，继续渲染画面
                        print(f"⏱️ 时间: {data.time:.2f}s | ❌ LTI 计算失败 (物理数据异常/奇异崩溃) | 错误: {e}")

                # 帧率控制
                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0: time.sleep(time_until_next)

            # ==== 在 while 循环正常结束 (即到达 duration) 后打印极值 ====
            print("\n" + "=" * 45)
            print(f"🏁 1 个正弦周期测试完成！")
            print(f"📈 平台最高点 (Max Z): {max_z:.4f} m")
            print(f"📉 平台最低点 (Min Z): {min_z:.4f} m")
            print(f"📏 总活动行程 (Stroke): {max_z - min_z:.4f} m")
            print("=" * 45 + "\n")
            # ==========================================
            # 📊 提取性能指标并保存到 TinyDB
            # ==========================================
            # 这里我们假设机构的“总活动行程 (Stroke)”就是我们要评估的性能指标 V
            # 你也可以随时改成纯粹的最高点: performance_v = max_z
            performance_v = max_z - min_z

            # 将该分数存入 Lookup Table
            # save_simulation_to_tinydb(xml_path, performance_v)
    except Exception as e:
        print(f"⚠️ 报错或切换至无头模式: {e}")


import os

# ==========================================
# 主程序入口 (交互式终端文件浏览器 + 批量/单步仿真)
# ==========================================
import time


def select_and_run():
    # 默认从项目的 data 文件夹开始，如果找不到则从当前目录(.)开始
    current_dir = "data" if os.path.exists("data") else "."

    while True:
        abs_path = os.path.abspath(current_dir)
        print("\n" + "=" * 60)
        print(f"📂 当前所在路径: {abs_path}")
        print("=" * 60)

        try:
            items = os.listdir(current_dir)
        except Exception as e:
            print(f"❌ 无法访问该目录: {e}")
            current_dir = os.path.dirname(current_dir)
            continue

        # 分类并过滤隐藏文件
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('.')])
        xml_files = sorted([f for f in items if f.endswith('.xml')])

        options = []
        options.append(("返回上一级", "..", "dir"))

        for d in dirs:
            options.append((f"📁 {d}", d, "dir"))
        for f in xml_files:
            options.append((f"🤖 {f}", f, "file"))

        # 打印菜单
        print(f"  [0] 🔙 返回上一级")

        # ⭐ 核心新增：如果当前目录下有 XML 文件，提供“一键全部播放”选项
        if xml_files:
            print(f"  [A] 🚀 自动按序仿真本目录下的所有 XML 文件 (共 {len(xml_files)} 个)")
            print("-" * 60)

        for i in range(1, len(options)):
            label, name, item_type = options[i]
            print(f"  [{i}] {label}")

        print("-" * 60)

        choice = input(f"👉 请输入编号或指令 (输入 'q' 退出): ").strip().lower()

        if choice == 'q':
            print("👋 已退出测试工具。")
            break

        # ==========================================
        # ⭐ 核心新增：批量按序仿真逻辑
        # ==========================================
        if choice == 'a' and xml_files:
            print(f"\n" + "🌟" * 20)
            print(f" 🚀 开始批量连续仿真，总计 {len(xml_files)} 个模型！")
            print(" 💡 提示：在仿真窗口中按 ESC 或点击关闭按钮，即可自动播放下一个。")
            print("🌟" * 20)
            time.sleep(2)

            for idx, xml_name in enumerate(xml_files, 1):
                selected_xml_path = os.path.join(current_dir, xml_name)

                print(f"\n" + "-" * 50)
                print(f" 🎯 正在播放 [{idx}/{len(xml_files)}]: {xml_name}")
                print("-" * 50)

                print(f"🧮 正在解析 XML 并计算自由度...")
                dof = compute_screw_theory_dof(selected_xml_path, run_sim=False)

                if dof <= 0:
                    print(f" ❌ 分析结果: 该机构为 死锁 或 过约束 (DOF={dof})")
                elif dof > 6:
                    print(f" ⚠️ 分析结果: 该机构为 欠约束/散架 状态 (DOF={dof})")
                else:
                    print(f" ✅ 分析结果: 该机构运转良好，输出自由度为 DOF = {dof}")

                # 短暂停顿，让终端输出能看清
                time.sleep(0.5)

                print(f"🚀 启动物理引擎...")
                # 运行仿真，此函数会阻塞，直到你关闭 MuJoCo 窗口
                test_parallel_robot(selected_xml_path)

            print("\n" + "🎉" * 20)
            print(" ✅ 当前文件夹下所有模型均已仿真完毕！")
            input("↩️ 请按【回车键】返回主菜单...")
            continue

        # ==========================================
        # 原有的单文件 / 文件夹跳转逻辑
        # ==========================================
        try:
            idx = int(choice)
            if 0 <= idx < len(options):
                _, name, item_type = options[idx]

                if idx == 0:
                    current_dir = os.path.dirname(abs_path)
                elif item_type == "dir":
                    current_dir = os.path.join(current_dir, name)
                elif item_type == "file":
                    selected_xml_path = os.path.join(current_dir, name)

                    while True:
                        print(f"\n" + "-" * 40)
                        print(f" 🎯 已单独选中: {name}")
                        print("-" * 40)

                        print(f"\n🧮 正在预先解析 XML 并计算自由度...")
                        dof = compute_screw_theory_dof(selected_xml_path, run_sim=False)

                        print("\n" + "✨" * 20)
                        if dof <= 0:
                            print(f" ❌ 分析结果: 该机构为 死锁 或 过约束 (DOF={dof})")
                        elif dof > 6:
                            print(f" ⚠️ 分析结果: 该机构为 欠约束/散架 状态 (DOF={dof})")
                        else:
                            print(f" ✅ 分析结果: 该机构运转良好，输出自由度为 DOF = {dof}")
                        print("✨" * 20)

                        time.sleep(1)

                        print(f"\n🚀 启动物理引擎: {selected_xml_path}")
                        test_parallel_robot(selected_xml_path)

                        input("\n↩️ 单次仿真结束，请按【回车键】返回目录...")
                        break

            else:
                print("❌ 无效的编号，请看清楚列表哦。")
        except ValueError:
            print("❌ 只能输入数字编号、'a' 或 'q' 呀。")


if __name__ == "__main__":
    select_and_run()