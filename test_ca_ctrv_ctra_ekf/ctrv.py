import numpy as np
class CTRVModel:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.dt = dt

    def state_transition_function(self, state, dt):
        x, y, v, theta, omega = state
        if omega != 0:
            x += v / omega * (np.sin(omega * dt + theta) - np.sin(theta))
            y += v / omega * (-np.cos(omega * dt + theta) + np.cos(theta))
        else:
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
        theta += omega * dt
        return np.array([x, y, v, theta, omega])

    def state_transition_jacobian(self, state, dt):
        x, y, v, theta, omega = state
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        if omega != 0:
            F = np.array([
                [1, 0, (np.sin(omega * dt + theta) - np.sin(theta)) / omega, v / omega * np.cos(omega * dt + theta) * dt, -v / omega**2 * (np.sin(omega * dt + theta) - np.sin(theta))],
                [0, 1, (-np.cos(omega * dt + theta) + np.cos(theta)) / omega, v / omega * np.sin(omega * dt + theta) * dt, -v / omega**2 * (-np.cos(omega * dt + theta) + np.cos(theta))],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, dt],
                [0, 0, 0, 0, 1]
            ])
        else:
            F = np.array([
                [1, 0, cos_theta * dt, -v * sin_theta * dt, 0],
                [0, 1, sin_theta * dt, v * cos_theta * dt, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, dt],
                [0, 0, 0, 0, 1]
            ])
        return F