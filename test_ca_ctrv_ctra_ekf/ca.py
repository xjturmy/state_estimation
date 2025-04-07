import numpy as np
class CAModel:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.dt = dt

    def state_transition_function(self, state, dt):
        x, y, v, a, theta = state
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        x += vx * dt
        y += vy * dt
        v += a * dt
        theta += 0.2 * dt  # 假设固定角速度
        a = 3.0 * np.cos(0.2 * dt) * 0.95  # 假设加速度变化
        return np.array([x, y, v, a, theta])

    def state_transition_jacobian(self, state, dt):
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