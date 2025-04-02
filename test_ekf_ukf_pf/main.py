import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter  # 假设 ExtendedKalmanFilter 类已经定义在 ekf 模块中
from ukf import UnscentedKalmanFilter  # 假设 UnscentedKalmanFilter 类已经定义在 ukf 模块中
from pf import ParticleFilter  # 假设 ParticleFilter 类已经定义在 pf 模块中
import time  # 导入 time 模块用于耗时统计

# 状态转移函数
def state_transition_function(state, dt):
    x, y, v, a, theta = state
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    x += vx * dt
    y += vy * dt
    v += a * dt
    omega = 0.2
    theta += omega * dt
    amplitude = 3.0
    decay_factor = 0.95
    a = amplitude * np.cos(omega * dt) * decay_factor
    return np.array([x, y, v, a, theta])

# 状态转移雅可比矩阵
def state_transition_jacobian(state, dt):
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

# 测量函数
def measurement_function(state):
    x, y, _, _, _ = state
    return np.array([x, y])

# 测量雅可比矩阵
def measurement_jacobian(state):
    H = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ])
    return H

# 初始化参数
dt = 1.0
num_steps = 50

# 真实状态初始化
true_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0])

# 初始化 EKF
ekf_initial_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0])
ekf_initial_covariance = np.eye(5) * 0.1
ekf_process_noise_cov = np.eye(5) * 0.01
ekf_measurement_noise_cov = np.eye(2) * 1.0
ekf = ExtendedKalmanFilter(ekf_initial_state, ekf_initial_covariance, ekf_process_noise_cov, ekf_measurement_noise_cov, dt)

# 初始化 UKF
ukf_initial_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0])
ukf_initial_covariance = np.eye(5) * 0.1
ukf_process_noise_cov = np.eye(5) * 0.01
ukf_measurement_noise_cov = np.eye(2) * 1.0
ukf = UnscentedKalmanFilter(ukf_initial_state, ukf_initial_covariance, ukf_process_noise_cov, ukf_measurement_noise_cov, dt)

# 初始化 PF
pf_initial_state = np.array([0.0, 0.0, 1.0, 0.1, 0.0])
pf_initial_covariance = np.eye(5) * 0.1
pf_process_noise_cov = np.eye(5) * 0.01
pf_measurement_noise_cov = np.eye(2) * 1.0
pf = ParticleFilter(1000, pf_initial_state, pf_initial_covariance, pf_process_noise_cov, pf_measurement_noise_cov, dt)

# 初始化状态估计列表
true_states = [true_state.copy()]
ekf_state_estimates = [ekf_initial_state.copy()]
ukf_state_estimates = [ukf_initial_state.copy()]
pf_state_estimates = [pf_initial_state.copy()]

# 初始化耗时列表
ekf_times = []
ukf_times = []
pf_times = []

# 模拟过程
for step in range(num_steps):
    # 真实状态更新
    true_state = state_transition_function(true_state, dt)
    true_states.append(true_state.copy())
    measurement = measurement_function(true_state) + np.random.normal(0, 1.0, 2)

    # EKF 预测和更新
    start_time = time.time()
    ekf.predict(state_transition_function, state_transition_jacobian)
    ekf.update(measurement, measurement_function, measurement_jacobian)
    ekf_state_estimates.append(ekf.get_state().copy())
    ekf_times.append(time.time() - start_time)

    # UKF 预测和更新
    start_time = time.time()
    ukf.predict(state_transition_function)
    ukf.update(measurement, measurement_function)
    ukf_state_estimates.append(ukf.state.copy())
    ukf_times.append(time.time() - start_time)

    # PF 预测和更新
    start_time = time.time()
    pf.predict(state_transition_function)
    pf.update(measurement, measurement_function)
    pf_state_estimates.append(pf.get_state())
    pf_times.append(time.time() - start_time)

# 将数据保存到 CSV 文件
true_states = np.array(true_states)
ekf_state_estimates = np.array(ekf_state_estimates)
ukf_state_estimates = np.array(ukf_state_estimates)
pf_state_estimates = np.array(pf_state_estimates)

# 只提取 x 和 y 的真实值、EKF 预测值、UKF 预测值和 PF 预测值
true_x_y = true_states[:, :2]
ekf_x_y = ekf_state_estimates[:, :2]
ukf_x_y = ukf_state_estimates[:, :2]
pf_x_y = pf_state_estimates[:, :2]

# 拼接数据
data = np.hstack((true_x_y, ekf_x_y, ukf_x_y, pf_x_y))

# 定义列名
columns = ['True_x', 'True_y', 'EKF_Est_x', 'EKF_Est_y', 'UKF_Est_x', 'UKF_Est_y', 'PF_Est_x', 'PF_Est_y']

# 创建 DataFrame
df = pd.DataFrame(data, columns=columns)

# 保存到 CSV 文件
df.to_csv('/home/gongyou/git_code/ruby_ws/src/state_estimation/test_ekf_ukf_pf/ekf_ukf_pf_data.csv', index=False)

print("数据已保存到 ekf_ukf_pf_data.csv 文件中。")

# 绘制轨迹图
plt.figure(figsize=(10, 8))
plt.plot(true_x_y[:, 0], true_x_y[:, 1], marker='o', linestyle='-', color='blue', label='True Trajectory')
plt.plot(ekf_x_y[:, 0], ekf_x_y[:, 1], marker='x', linestyle='--', color='green', label='EKF Estimate')
plt.plot(ukf_x_y[:, 0], ukf_x_y[:, 1], marker='x', linestyle='--', color='red', label='UKF Estimate')
plt.plot(pf_x_y[:, 0], pf_x_y[:, 1], marker='x', linestyle='--', color='purple', label='PF Estimate')
plt.title('Comparison of True Trajectory, EKF, UKF, and PF Estimates')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# 打印耗时统计结果
print("耗时统计（单位：毫秒）：")
print(f"EKF 平均耗时: {np.mean(ekf_times) * 1000:.6f}")
print(f"UKF 平均耗时: {np.mean(ukf_times) * 1000:.6f}")
print(f"PF 平均耗时: {np.mean(pf_times) * 1000:.6f}")