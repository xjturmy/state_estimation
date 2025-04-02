import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt, alpha=1e-3, beta=2, kappa=0):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n = initial_state.shape[0]
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        self.weights_mean, self.weights_covariance = self._calculate_weights()

    def _calculate_weights(self):
        weights_mean = np.zeros(2 * self.n + 1)
        weights_covariance = np.zeros(2 * self.n + 1)
        weights_mean[0] = self.lambda_ / (self.n + self.lambda_)
        weights_covariance[0] = weights_mean[0] + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2 * self.n + 1):
            weights_mean[i] = weights_covariance[i] = 1 / (2 * (self.n + self.lambda_))
        return weights_mean, weights_covariance

    def _generate_sigma_points(self):
        sqrt_matrix = np.linalg.cholesky((self.n + self.lambda_) * self.covariance)
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state
        for i in range(self.n):
            sigma_points[i + 1] = self.state + self.gamma * sqrt_matrix[:, i]
            sigma_points[self.n + i + 1] = self.state - self.gamma * sqrt_matrix[:, i]
        return sigma_points

    def predict(self, state_transition_function):
        sigma_points = self._generate_sigma_points()
        predicted_sigma_points = np.array([state_transition_function(point, self.dt) for point in sigma_points])
        self.state = np.dot(self.weights_mean, predicted_sigma_points)
        self.covariance = self.process_noise_cov.copy()
        for i in range(2 * self.n + 1):
            diff = predicted_sigma_points[i] - self.state
            self.covariance += self.weights_covariance[i] * np.outer(diff, diff)

    def update(self, measurement, measurement_function):
        predicted_measurements = np.array([measurement_function(point) for point in self._generate_sigma_points()])
        predicted_measurement = np.dot(self.weights_mean, predicted_measurements)
        innovation_covariance = self.measurement_noise_cov.copy()
        for i in range(2 * self.n + 1):
            diff = predicted_measurements[i] - predicted_measurement
            innovation_covariance += self.weights_covariance[i] * np.outer(diff, diff)
        cross_correlation = np.zeros((self.n, 2))
        for i in range(2 * self.n + 1):
            diff_state = self._generate_sigma_points()[i] - self.state
            diff_measurement = predicted_measurements[i] - predicted_measurement
            cross_correlation += self.weights_covariance[i] * np.outer(diff_state, diff_measurement)
        kalman_gain = np.dot(cross_correlation, np.linalg.inv(innovation_covariance))
        self.state += np.dot(kalman_gain, (measurement - predicted_measurement))
        self.covariance -= np.dot(kalman_gain, np.dot(innovation_covariance, kalman_gain.T))