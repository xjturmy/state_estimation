import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt):
        """
        初始化扩展卡尔曼滤波器
        :param initial_state: 初始状态向量 [x, y, v, theta]
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

    def predict(self, f, F):
        """
        预测步骤
        :param f: 状态转移函数 f(state, dt)
        :param F: 状态转移雅可比矩阵 F(state, dt)
        """
        # 状态预测：x_new = f(x, dt)
        self.state = f(self.state, self.dt)
        # 协方差预测：P_new = F * P * F.T + Q
        self.P = F(self.state, self.dt) @ self.P @ F(self.state, self.dt).T + self.Q

    def update(self, measurement, h, H):
        """
        更新步骤
        :param measurement: 测量值 [x, y]
        :param h: 测量函数 h(state)
        :param H: 测量雅可比矩阵 H(state)
        """
        # 计算卡尔曼增益
        S = H(self.state) @ self.P @ H(self.state).T + self.R  # 预测误差协方差
        K = self.P @ H(self.state).T @ np.linalg.inv(S)  # 卡尔曼增益
        # 状态更新：x = x + K * (z - h(x))
        self.state = self.state + K @ (measurement - h(self.state))
        # 协方差更新：P = (I - K * H) * P
        self.P = (np.eye(len(self.state)) - K @ H(self.state)) @ self.P

    def get_state(self):
        """
        获取当前状态
        :return: 当前状态向量 [x, y, v, theta]
        """
        return self.state