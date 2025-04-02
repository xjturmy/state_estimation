import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 数据加载 ===
# 假设数据已经保存在 CSV 文件中，文件路径为 'data.csv'
data = pd.read_csv('/home/gongyou/git_code/ruby_ws/src/state_estimation/test_cv_kf_ekf/data.csv')  # 替换为你的数据文件路径

# === 误差计算 ===
# 计算估计误差（欧几里得距离）
kf_error = np.sqrt((data['True_x'] - data['KF_Est_x'])**2 + (data['True_y'] - data['KF_Est_y'])**2)
ekf_error = np.sqrt((data['True_x'] - data['EKF_Est_x'])**2 + (data['True_y'] - data['EKF_Est_y'])**2)

# 打印估计误差结果
print("估计误差（欧几里得距离）：")
print(f"KF 误差: {kf_error.mean():.3f}")
print(f"EKF 误差: {ekf_error.mean():.3f}")

# === 绘图 ===
# 创建一个图形窗口
plt.figure(figsize=(14, 12))

# 第一个子图：x 方向的位置估计
plt.subplot(2, 2, 1)  # 2行2列的第1个子图
plt.plot(data['True_x'], label='True Position (x)', marker='o', color='blue')
plt.plot(data['KF_Est_x'], label='KF Estimate (x)', linestyle='--', color='red')
plt.plot(data['EKF_Est_x'], label='EKF Estimate (x)', linestyle='-.', color='green')
plt.title('x Position Estimate')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.legend()
plt.grid(True)

# 第二个子图：y 方向的位置估计
plt.subplot(2, 2, 2)  # 2行2列的第2个子图
plt.plot(data['True_y'], label='True Position (y)', marker='o', color='blue')
plt.plot(data['KF_Est_y'], label='KF Estimate (y)', linestyle='--', color='red')
plt.plot(data['EKF_Est_y'], label='EKF Estimate (y)', linestyle='-.', color='green')
plt.title('y Position Estimate')
plt.xlabel('Iteration')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 第三个子图：x 方向的偏差
plt.subplot(2, 2, 3)  # 2行2列的第3个子图
plt.plot(data['True_x'] - data['KF_Est_x'], label='KF Error (x)', marker='x', linestyle='--', color='red')
plt.plot(data['True_x'] - data['EKF_Est_x'], label='EKF Error (x)', marker='x', linestyle='--', color='green')
plt.title('x Position Errors')
plt.xlabel('Iteration')
plt.ylabel('Error (x)')
plt.legend()
plt.grid(True)

# 第四个子图：y 方向的偏差
plt.subplot(2, 2, 4)  # 2行2列的第4个子图
plt.plot(data['True_y'] - data['KF_Est_y'], label='KF Error (y)', marker='x', linestyle='--', color='red')
plt.plot(data['True_y'] - data['EKF_Est_y'], label='EKF Error (y)', marker='x', linestyle='--', color='green')
plt.title('y Position Errors')
plt.xlabel('Iteration')
plt.ylabel('Error (y)')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()  # 自动调整子图间距
plt.show()