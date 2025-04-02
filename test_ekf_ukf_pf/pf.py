import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt):
        """
        初始化粒子滤波器
        :param num_particles: 粒子数量
        :param initial_state: 初始状态向量
        :param initial_covariance: 初始状态协方差矩阵
        :param process_noise_cov: 过程噪声协方差矩阵
        :param measurement_noise_cov: 测量噪声协方差矩阵
        :param dt: 时间步长
        """
        self.num_particles = num_particles
        self.particles = np.random.multivariate_normal(initial_state, initial_covariance, num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.dt = dt

    def predict(self, state_transition_function):
        """
        预测步骤
        :param state_transition_function: 状态转移函数
        """
        for i in range(self.num_particles):
            self.particles[i] = state_transition_function(self.particles[i], self.dt) + np.random.multivariate_normal(np.zeros_like(self.particles[i]), self.process_noise_cov)

    def update(self, measurement, measurement_function):
        """
        更新步骤
        :param measurement: 测量值
        :param measurement_function: 测量函数
        """
        for i in range(self.num_particles):
            predicted_measurement = measurement_function(self.particles[i])
            self.weights[i] *= np.exp(-0.5 * np.dot((measurement - predicted_measurement).T, np.linalg.solve(self.measurement_noise_cov, (measurement - predicted_measurement))))
        self.weights /= np.sum(self.weights)
        self.resample()

    def resample(self):
        """
        重采样步骤
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_state(self):
        """
        获取当前状态估计
        :return: 当前状态估计
        """
        return np.average(self.particles, weights=self.weights, axis=0)