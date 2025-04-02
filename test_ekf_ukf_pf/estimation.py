import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data = pd.read_csv('/home/gongyou/git_code/ruby_ws/src/state_estimation/test_ekf_ukf_pf/ekf_ukf_pf_data.csv')

# 计算EKF估计误差（欧几里得距离）
ekf_error = np.sqrt((data['True_x'] - data['EKF_Est_x'])**2 + (data['True_y'] - data['EKF_Est_y'])**2)

# 计算UKF估计误差（欧几里得距离）
ukf_error = np.sqrt((data['True_x'] - data['UKF_Est_x'])**2 + (data['True_y'] - data['UKF_Est_y'])**2)

# 计算PF估计误差（欧几里得距离）
pf_error = np.sqrt((data['True_x'] - data['PF_Est_x'])**2 + (data['True_y'] - data['PF_Est_y'])**2)

# 打印EKF、UKF和PF的估计误差结果
print("估计误差（欧几里得距离）：")
print(f"EKF 误差: {ekf_error.mean():.12f}")  # 保留12位有效数字
print(f"UKF 误差: {ukf_error.mean():.12f}")  # 保留12位有效数字
print(f"PF 误差: {pf_error.mean():.12f}")  # 保留12位有效数字

# 创建一个图形窗口
plt.figure(figsize=(14, 12))

# 第一个子图：x方向的位置估计
plt.subplot(2, 2, 1)  # 2行2列的第1个子图
plt.plot(data['True_x'], label='True Position (x)', marker='o', color='blue')
plt.plot(data['EKF_Est_x'], label='EKF Estimate (x)', linestyle='-.', color='green')
plt.plot(data['UKF_Est_x'], label='UKF Estimate (x)', linestyle='--', color='red')
plt.plot(data['PF_Est_x'], label='PF Estimate (x)', linestyle=':', color='purple')
plt.title('x Position Estimate')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.legend()
plt.grid(True)

# 第二个子图：y方向的位置估计
plt.subplot(2, 2, 2)  # 2行2列的第2个子图
plt.plot(data['True_y'], label='True Position (y)', marker='o', color='blue')
plt.plot(data['EKF_Est_y'], label='EKF Estimate (y)', linestyle='-.', color='green')
plt.plot(data['UKF_Est_y'], label='UKF Estimate (y)', linestyle='--', color='red')
plt.plot(data['PF_Est_y'], label='PF Estimate (y)', linestyle=':', color='purple')
plt.title('y Position Estimate')
plt.xlabel('Iteration')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 第三个子图：x方向的偏差
plt.subplot(2, 2, 3)  # 2行2列的第3个子图
plt.plot(data['True_x'] - data['EKF_Est_x'], label='EKF Error (x)', marker='x', linestyle='--', color='green')
plt.plot(data['True_x'] - data['UKF_Est_x'], label='UKF Error (x)', marker='x', linestyle='--', color='red')
plt.plot(data['True_x'] - data['PF_Est_x'], label='PF Error (x)', marker='x', linestyle='--', color='purple')
plt.title('x Position Errors')
plt.xlabel('Iteration')
plt.ylabel('Error (x)')
plt.legend()
plt.grid(True)

# 第四个子图：y方向的偏差
plt.subplot(2, 2, 4)  # 2行2列的第4个子图
plt.plot(data['True_y'] - data['EKF_Est_y'], label='EKF Error (y)', marker='x', linestyle='--', color='green')
plt.plot(data['True_y'] - data['UKF_Est_y'], label='UKF Error (y)', marker='x', linestyle='--', color='red')
plt.plot(data['True_y'] - data['PF_Est_y'], label='PF Error (y)', marker='x', linestyle='--', color='purple')
plt.title('y Position Errors')
plt.xlabel('Iteration')
plt.ylabel('Error (y)')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()  # 自动调整子图间距
plt.show()

# 保存误差数据到 CSV 文件，保留更多有效数字
error_data = np.hstack((ekf_error[:, np.newaxis], ukf_error[:, np.newaxis], pf_error[:, np.newaxis]))
error_columns = ['EKF_Error', 'UKF_Error', 'PF_Error']
error_df = pd.DataFrame(error_data, columns=error_columns)
error_df.to_csv('/home/gongyou/git_code/ruby_ws/src/python/EKF_UKF_PF/ekf_ukf_pf_errors.csv', index=False, float_format='%.12f')  # 保留12位有效数字