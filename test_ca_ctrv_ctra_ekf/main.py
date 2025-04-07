import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter  # 假设 ExtendedKalmanFilter 类已经定义在 ekf 模块中
from ctra import CTRAModel  # 假设 CTRAModel 类已经定义在 ctra 模块中
from ctrv import CTRVModel  # 假设 CTRVModel 类已经定义在 ctrv 模块中
from ca import CAModel  # 假设 CAModel 类已经定义在 ca 模块中
import time  # 导入时间模块

# 测量函数
def measurement_function_ca(state):
    x, y, _, _, _ = state
    return np.array([x, y])

def measurement_function_ctrv(state):
    x, y, _, _, _ = state
    return np.array([x, y])

def measurement_function_ctra(state):
    x, y, _, _, _, _ = state  # 解包 6 个值
    return np.array([x, y])

# 测量雅可比矩阵
def measurement_jacobian_ca(state):
    H = np.array([
        [1, 0, 0, 0, 0],  # 测量 x
        [0, 1, 0, 0, 0]   # 测量 y
    ])
    return H

def measurement_jacobian_ctrv(state):
    H = np.array([
        [1, 0, 0, 0, 0],  # 测量 x
        [0, 1, 0, 0, 0]   # 测量 y
    ])
    return H

def measurement_jacobian_ctra(state):
    H = np.array([
        [1, 0, 0, 0, 0, 0],  # 测量 x
        [0, 1, 0, 0, 0, 0]   # 测量 y
    ])
    return H

# 读取CSV文件中的真实数据
data = pd.read_csv('/home/gongyou/git_code/state_estimation/test_ca_ctrv_ctra_ekf/rectangles.csv')
true_x = data['Center_X'].values
true_y = data['Center_Y'].values
num_steps = len(true_x)

# 初始化 CA 模型的 EKF
ca_model = CAModel(
    initial_state=np.array([true_x[0], true_y[0], 1.0, 0.1, 0.0]),
    initial_covariance=np.eye(5) * 0.1,
    process_noise_cov=np.eye(5) * 0.01,
    measurement_noise_cov=np.eye(2) * 1.0,
    dt=1.0
)
ca_ekf = ExtendedKalmanFilter(
    ca_model.state,
    ca_model.covariance,
    ca_model.process_noise_cov,
    ca_model.measurement_noise_cov,
    dt=ca_model.dt
)

# 初始化 CTRV 模型的 EKF
ctrv_model = CTRVModel(
    initial_state=np.array([true_x[0], true_y[0], 1.0, 0.0, 0.1]),
    initial_covariance=np.eye(5) * 0.1,
    process_noise_cov=np.eye(5) * 0.01,
    measurement_noise_cov=np.eye(2) * 1.0,
    dt=1.0
)
ctrv_ekf = ExtendedKalmanFilter(
    ctrv_model.state,
    ctrv_model.covariance,
    ctrv_model.process_noise_cov,
    ctrv_model.measurement_noise_cov,
    dt=ctrv_model.dt
)

# 初始化 CTRA 模型的 EKF
ctra_model = CTRAModel(
    initial_state=np.array([true_x[0], true_y[0], 1.0, 0.0, 0.1, 0.1]),
    initial_covariance=np.eye(6) * 0.1,
    process_noise_cov=np.eye(6) * 0.01,
    measurement_noise_cov=np.eye(2) * 1.0,
    dt=1.0
)
ctra_ekf = ExtendedKalmanFilter(
    ctra_model.state,
    ctra_model.covariance,
    ctra_model.process_noise_cov,
    ctra_model.measurement_noise_cov,
    dt=ctra_model.dt
)

# 初始化状态估计列表
ca_ekf_state_estimates = [ca_model.state.copy()]
ctrv_ekf_state_estimates = [ctrv_model.state.copy()]
ctra_ekf_state_estimates = [ctra_model.state.copy()]

# 初始化耗时列表
ca_times = []
ctrv_times = []
ctra_times = []

# 模拟过程
for step in range(num_steps):
    # 真实状态更新
    measurement = np.array([true_x[step], true_y[step]]) + np.random.normal(0, 1.0, 2)  # 添加测量噪声

    # CA 模型的 EKF 预测和更新
    start_time = time.time()
    ca_ekf.predict(ca_model.state_transition_function, ca_model.state_transition_jacobian)
    ca_ekf.update(measurement, measurement_function_ca, measurement_jacobian_ca)
    ca_ekf_state_estimates.append(ca_ekf.get_state().copy())
    ca_times.append(time.time() - start_time)

    # CTRV 模型的 EKF 预测和更新
    start_time = time.time()
    ctrv_ekf.predict(ctrv_model.state_transition_function, ctrv_model.state_transition_jacobian)
    ctrv_ekf.update(measurement, measurement_function_ctrv, measurement_jacobian_ctrv)
    ctrv_ekf_state_estimates.append(ctrv_ekf.get_state().copy())
    ctrv_times.append(time.time() - start_time)

    # CTRA 模型的 EKF 预测和更新
    start_time = time.time()
    ctra_ekf.predict(ctra_model.state_transition_function, ctra_model.state_transition_jacobian)
    ctra_ekf.update(measurement, measurement_function_ctra, measurement_jacobian_ctra)
    ctra_ekf_state_estimates.append(ctra_ekf.get_state().copy())
    ctra_times.append(time.time() - start_time)

# 移除初始状态的重复
ca_ekf_state_estimates = np.array(ca_ekf_state_estimates[1:])  # 移除第一个重复的初始状态
ctrv_ekf_state_estimates = np.array(ctrv_ekf_state_estimates[1:])  # 移除第一个重复的初始状态
ctra_ekf_state_estimates = np.array(ctra_ekf_state_estimates[1:])  # 移除第一个重复的初始状态

# 输出三种方法的平均耗时（以毫秒为单位）
print("平均耗时：")
print(f"CA EKF 耗时: {np.mean(ca_times) * 1000:.6f} 毫秒")
print(f"CTRV EKF 耗时: {np.mean(ctrv_times) * 1000:.6f} 毫秒")
print(f"CTRA EKF 耗时: {np.mean(ctra_times) * 1000:.6f} 毫秒")

# 将数据保存到 CSV 文件
true_states = np.vstack((true_x, true_y)).T
ca_ekf_state_estimates = np.array(ca_ekf_state_estimates)
ctrv_ekf_state_estimates = np.array(ctrv_ekf_state_estimates)
ctra_ekf_state_estimates = np.array(ctra_ekf_state_estimates)

# 提取 x 和 y 的真实值、CA 模型、CTRV 模型和 CTRA 模型的预测值
true_x_y = true_states[:, :2]
ca_ekf_x_y = ca_ekf_state_estimates[:, :2]
ctrv_ekf_x_y = ctrv_ekf_state_estimates[:, :2]
ctra_ekf_x_y = ctra_ekf_state_estimates[:, :2]

# 拼接数据
data = np.hstack((true_x_y, ca_ekf_x_y, ctrv_ekf_x_y, ctra_ekf_x_y))

# 定义列名
columns = ['True_x', 'True_y', 'CA_EKF_Est_x', 'CA_EKF_Est_y', 'CTRV_EKF_Est_x', 'CTRV_EKF_Est_y', 'CTRA_EKF_Est_x', 'CTRA_EKF_Est_y']

# 创建 DataFrame
df = pd.DataFrame(data, columns=columns)

# 保存到 CSV 文件
df.to_csv('/home/gongyou/git_code/state_estimation/test_ca_ctrv_ctra_ekf/ca_ctrv_ctra_data.csv', index=False)

print("数据已保存到 ca_ctrv_ctra_data.csv 文件中。")

# 绘制轨迹图
plt.figure(figsize=(10, 8))
plt.plot(true_x_y[:, 0], true_x_y[:, 1], marker='o', linestyle='-', color='blue', label='True Trajectory')
plt.plot(ca_ekf_x_y[:, 0], ca_ekf_x_y[:, 1], marker='x', linestyle='--', color='green', label='CA EKF Estimate')
plt.plot(ctrv_ekf_x_y[:, 0], ctrv_ekf_x_y[:, 1], marker='x', linestyle='--', color='red', label='CTRV EKF Estimate')
plt.plot(ctra_ekf_x_y[:, 0], ctra_ekf_x_y[:, 1], marker='x', linestyle='--', color='purple', label='CTRA EKF Estimate')
plt.title('Comparison of True Trajectory, CA EKF, CTRV EKF, and CTRA EKF Estimates')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保持比例
plt.show()