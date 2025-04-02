import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt):
        """
        初始化卡尔曼滤波器
        :param initial_state: 初始状态向量 [x, y, vx, vy]
        :param initial_covariance: 初始状态协方差矩阵
        :param process_noise_cov: 过程噪声协方差矩阵
        :param measurement_noise_cov: 测量噪声协方差矩阵
        :param dt: 时间步长
        """
        self.state = initial_state  # 当前状态向量
        self.P = initial_covariance  # 当前状态协方差矩阵
        self.Q = process_noise_cov  # 过程噪声协方差矩阵
        self.R = measurement_noise_cov  # 测量噪声协方差矩阵
        self.dt = dt  # 时间步长

        # 状态转移矩阵，假设为匀速运动模型
        self.F = np.array([
            [1, 0, dt, 0],  # x 的更新：x_new = x + vx * dt
            [0, 1, 0, dt],  # y 的更新：y_new = y + vy * dt
            [0, 0, 1, 0],   # vx 的更新：vx_new = vx
            [0, 0, 0, 1]    # vy 的更新：vy_new = vy
        ])

        # 测量矩阵，只测量位置信息 (x 和 y)
        self.H = np.array([
            [1, 0, 0, 0],  # 测量 x
            [0, 1, 0, 0]   # 测量 y
        ])

    def predict(self):
        """
        预测步骤
        """
        # 状态预测：x_new = F * x
        self.state = self.F @ self.state
        # 协方差预测：P_new = F * P * F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        更新步骤
        :param measurement: 测量值 [x, y]
        """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R  # 预测误差协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        # 状态更新：x = x + K * (z - H * x)
        self.state = self.state + K @ (measurement - self.H @ self.state)
        # 协方差更新：P = (I - K * H) * P
        self.P = (np.eye(len(self.state)) - K @ self.H) @ self.P

    def get_state(self):
        """
        获取当前状态
        :return: 当前状态向量 [x, y, vx, vy]
        """
        return self.state