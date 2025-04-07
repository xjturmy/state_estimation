import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data = pd.read_csv('/home/gongyou/git_code/state_estimation/test_ca_ctrv_ctra_ekf/ca_ctrv_ctra_data.csv')

# 计算每个模型的 EKF 估计误差（欧几里得距离）
ca_ekf_error = np.sqrt((data['True_x'] - data['CA_EKF_Est_x'])**2 + (data['True_y'] - data['CA_EKF_Est_y'])**2)
ctrv_ekf_error = np.sqrt((data['True_x'] - data['CTRV_EKF_Est_x'])**2 + (data['True_y'] - data['CTRV_EKF_Est_y'])**2)
ctra_ekf_error = np.sqrt((data['True_x'] - data['CTRA_EKF_Est_x'])**2 + (data['True_y'] - data['CTRA_EKF_Est_y'])**2)

# 打印每个模型的估计误差结果
print("估计误差（欧几里得距离）：")
print(f"CA EKF 误差: {ca_ekf_error.mean():.3f}")
print(f"CTRV EKF 误差: {ctrv_ekf_error.mean():.3f}")
print(f"CTRA EKF 误差: {ctra_ekf_error.mean():.3f}")

# 创建一个图形窗口
plt.figure(figsize=(14, 12))

# 第一个子图：x方向的位置估计
plt.subplot(2, 2, 1)  # 2行2列的第1个子图
plt.plot(data['True_x'], label='True Position (x)', marker='o', color='blue')
plt.plot(data['CA_EKF_Est_x'], label='CA EKF Estimate (x)', linestyle='-.', color='green')
plt.plot(data['CTRV_EKF_Est_x'], label='CTRV EKF Estimate (x)', linestyle='--', color='red')
plt.plot(data['CTRA_EKF_Est_x'], label='CTRA EKF Estimate (x)', linestyle=':', color='purple')
plt.title('x Position Estimate')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.legend()
plt.grid(True)

# 第二个子图：y方向的位置估计
plt.subplot(2, 2, 2)  # 2行2列的第2个子图
plt.plot(data['True_y'], label='True Position (y)', marker='o', color='blue')
plt.plot(data['CA_EKF_Est_y'], label='CA EKF Estimate (y)', linestyle='-.', color='green')
plt.plot(data['CTRV_EKF_Est_y'], label='CTRV EKF Estimate (y)', linestyle='--', color='red')
plt.plot(data['CTRA_EKF_Est_y'], label='CTRA EKF Estimate (y)', linestyle=':', color='purple')
plt.title('y Position Estimate')
plt.xlabel('Iteration')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 第三个子图：x方向的偏差
plt.subplot(2, 2, 3)  # 2行2列的第3个子图
plt.plot(data['True_x'] - data['CA_EKF_Est_x'], label='CA EKF Error (x)', marker='x', linestyle='--', color='green')
plt.plot(data['True_x'] - data['CTRV_EKF_Est_x'], label='CTRV EKF Error (x)', marker='x', linestyle='--', color='red')
plt.plot(data['True_x'] - data['CTRA_EKF_Est_x'], label='CTRA EKF Error (x)', marker='x', linestyle='--', color='purple')
plt.title('x Position Errors')
plt.xlabel('Iteration')
plt.ylabel('Error (x)')
plt.legend()
plt.grid(True)

# 第四个子图：y方向的偏差
plt.subplot(2, 2, 4)  # 2行2列的第4个子图
plt.plot(data['True_y'] - data['CA_EKF_Est_y'], label='CA EKF Error (y)', marker='x', linestyle='--', color='green')
plt.plot(data['True_y'] - data['CTRV_EKF_Est_y'], label='CTRV EKF Error (y)', marker='x', linestyle='--', color='red')
plt.plot(data['True_y'] - data['CTRA_EKF_Est_y'], label='CTRA EKF Error (y)', marker='x', linestyle='--', color='purple')
plt.title('y Position Errors')
plt.xlabel('Iteration')
plt.ylabel('Error (y)')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()  # 自动调整子图间距
plt.show()