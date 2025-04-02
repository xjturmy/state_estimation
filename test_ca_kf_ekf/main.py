import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kf_accel import KalmanFilterAccel  # 假设 KalmanFilterAccel 类已经定义在 kf_accel 模块中
from ekf_accel import ExtendedKalmanFilter  # 假设 ExtendedKalmanFilter 类已经定义在 ekf_accel 模块中

# === 状态转移和测量模型 ===
def state_transition_function(state, dt):
    """
    状态转移函数 f(state, dt)
    """
    x, y, v, a, theta = state  # 当前状态向量 [x, y, v, a, theta]

    # 更新水平位置和速度
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    # 更新位置
    x += vx * dt
    y += vy * dt

    # 更新速度
    v += a * dt

    # 更新方向角 theta，使其随时间变化，模拟波浪运动
    omega = 0.2  # 角频率，增加到0.2
    theta += omega * dt

    # 模拟波浪运动的加速度变化
    amplitude = 3.0  # 振幅，增加到3.0
    decay_factor = 0.95  # 衰减因子，调整为0.95
    a = amplitude * np.cos(omega * dt) * decay_factor

    # 返回更新后的状态向量 [x, y, v, a, theta]
    return np.array([x, y, v, a, theta])

def state_transition_jacobian(state, dt):
    """
    状态转移雅可比矩阵 F(state, dt)
    """
    x, y, v, a, theta = state
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    F = np.array([
        [1, 0, cos_theta * dt, 0.5 * cos_theta * dt**2, -v * sin_theta * dt],
        [0, 1, sin_theta * dt, 0.5 * sin_theta * dt**2, v * cos_theta * dt],
        [0, 0, 1, dt, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    return F

def measurement_function(state):
    """
    测量函数 h(state)
    """
    x, y, _, _, _ = state
    return np.array([x, y])

def measurement_jacobian(state):
    """
    测量雅可比矩阵 H(state)
    """
    H = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ])
    return H

# === 滤波器初始化 ===
def initialize_filters(dt):
    """
    初始化 KF 和 EKF
    """
    # 初始化 EKF
    ekf_initial_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0])  # 初始状态 [x, y, v, a, theta]
    ekf_initial_covariance = np.eye(5) * 0.1  # 初始协方差矩阵
    ekf_process_noise_cov = np.eye(5) * 0.01  # 过程噪声协方差矩阵
    ekf_measurement_noise_cov = np.eye(2) * 1.0  # 测量噪声协方差矩阵
    ekf = ExtendedKalmanFilter(ekf_initial_state, ekf_initial_covariance, ekf_process_noise_cov, ekf_measurement_noise_cov, dt)

    # 初始化 KF
    kf_initial_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0, 0.0])  # 初始状态 [x, y, vx, vy, ax, ay]
    kf_initial_covariance = np.eye(6) * 0.1  # 初始协方差矩阵
    kf_process_noise_cov = np.eye(6) * 0.01  # 过程噪声协方差矩阵
    kf_measurement_noise_cov = np.eye(2) * 1.0  # 测量噪声协方差矩阵
    kf = KalmanFilterAccel(kf_initial_state, kf_initial_covariance, kf_process_noise_cov, kf_measurement_noise_cov, dt)

    return kf, ekf

# === 模拟过程 ===
def run_simulation(dt, num_steps):
    """
    执行模拟过程
    """
    # 初始化真实状态
    true_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0])  # [x, y, v, a, theta]
    true_states = [true_state.copy()]

    # 初始化滤波器
    kf, ekf = initialize_filters(dt)

    # 初始化状态估计列表
    ekf_state_estimates = [ekf.get_state().copy()]
    kf_state_estimates = [kf.get_state().copy()]

    # 模拟过程
    for _ in range(num_steps):
        # 真实状态更新
        true_state = state_transition_function(true_state, dt)
        true_states.append(true_state.copy())

        # EKF 预测和更新
        ekf.predict(state_transition_function, state_transition_jacobian)
        measurement = measurement_function(true_state) + np.random.normal(0, 1.0, 2)  # 添加测量噪声
        ekf.update(measurement, measurement_function, measurement_jacobian)
        ekf_state_estimates.append(ekf.get_state().copy())

        # KF 预测和更新
        kf.predict()
        kf.update(measurement)
        kf_state_estimates.append(kf.get_state().copy())

    return np.array(true_states), np.array(kf_state_estimates), np.array(ekf_state_estimates)

# === 数据保存和绘图 ===
if __name__ == "__main__":
    # 初始化参数
    dt = 1.0  # 时间步长
    num_steps = 50  # 模拟步数

    # 运行模拟
    true_states, kf_state_estimates, ekf_state_estimates = run_simulation(dt, num_steps)

    # 只提取 x 和 y 的真实值、EKF 预测值和 KF 预测值
    true_x_y = true_states[:, :2]  # 真实状态的 x 和 y
    ekf_x_y = ekf_state_estimates[:, :2]  # EKF 预测状态的 x 和 y
    kf_x_y = kf_state_estimates[:, :2]  # KF 预测状态的 x 和 y

    # 拼接数据
    data = np.hstack((true_x_y, ekf_x_y, kf_x_y))

    # 定义列名
    columns = ['True_x', 'True_y', 'EKF_Est_x', 'EKF_Est_y', 'KF_Est_x', 'KF_Est_y']

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 保存到 CSV 文件
    df.to_csv('/home/gongyou/git_code/ruby_ws/src/state_estimation/test_ca_kf_ekf/data.csv', index=False)
    print("数据已保存到 data.csv 文件中。")

    # 绘制轨迹图
    plt.figure(figsize=(10, 8))
    plt.plot(true_x_y[:, 0], true_x_y[:, 1], marker='o', linestyle='-', color='blue', label='True Trajectory')
    plt.plot(ekf_x_y[:, 0], ekf_x_y[:, 1], marker='x', linestyle='--', color='green', label='EKF Estimate')
    plt.plot(kf_x_y[:, 0], kf_x_y[:, 1], marker='x', linestyle='--', color='red', label='KF Estimate')
    plt.title('Comparison of True Trajectory, EKF, and KF Estimates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保持比例
    plt.show()