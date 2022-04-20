from scipy.io import loadmat
import numpy as np
import torch


class RealisticMeasModel:
    def __init__(self):
        data = loadmat('src/results/precomputed_covs/precomputed_covs.mat')
        self.y = torch.tensor(data['Y'])
        self.min_r = min(data['X'][:, 0])
        self.max_r = max(data['X'][:, 0])
        self.min_theta = min(data['X'][:, 1])
        self.max_theta = max(data['X'][:, 1])

        self.r_delta = data['range_delta'].item()
        self.theta_delta = data['theta_delta'].item()
        self.n_r = data['Nrange'].item()
        self.n_theta = data['Ntheta'].item()

        # Permute column and rows so that covariances are for [r, doppler, theta] (instead of [r, theta, doppler]
        self.y = self.y[:, [0, 2, 1]][:, :, [0, 2, 1]]

    def compute_covariance(self, x):
        """
        Looks up the closest covariance matrix to the point `x` given.
        @param x: `ndarray` of shape [N, 2], specifying (range, doppler, theta) of the N points where we want to compute
            the covariance matrices (note that the doppler part does not influence the covariances).
        @return: `ndarray` of shape [N, 3, 3] with the covariance matrices computed for each of the N points in `x`. The
            covariances are specified for (range, doppler, theta), in this order.
        """
        n_points = x.shape[0]
        if n_points == 0:
            return np.zeros((0, 3, 3))
        i_r = np.clip(np.round((x[:, 0] - self.min_r) / self.r_delta), 0, self.n_r - 1)
        i_theta = np.clip(np.round((x[:, 2] - self.min_theta) / self.theta_delta), 0, self.n_theta - 1)
        idx = (i_r*self.n_theta + i_theta).astype(int)
        return self.y[idx]


# Example usage
if __name__ == '__main__':
    # Create measurement model
    model = RealisticMeasModel()

    # Query at two points
    point = np.array([[16, 13, 0.3], [4, -2, 1.7]])
    y = model.compute_covariance(point)

    # Print results
    print(y)
