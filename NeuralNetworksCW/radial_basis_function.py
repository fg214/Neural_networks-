import numpy as np
from scipy.linalg import pinv
from scipy.spatial.distance import euclidean, cdist


class RadialBasisFunctionNeuralNetwork:

    def __init__(self, num_centers, training_inputs, training_outputs, regularization_term, std_multiplier):
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.number_of_centers = num_centers
        self.regularization_term = regularization_term
        self.std_multiplier = std_multiplier
        self.centers, self.closest_center_index = self.k_means(num_centers)
        self.sigmas = self._calc_sigmas(self.centers, self.closest_center_index)
        self.weights = self.fit()
        print("With {} centers".format(num_centers))

    @staticmethod
    def gaussian_radial_basis_function(feature_vector, centers_sigma):
        return [np.exp(-(((euclidean(center, feature_vector)) ** 2) / (2 * (sigma ** 2)))) for center, sigma in
                centers_sigma]

    def k_means(self, num_centers):
        training_feature_vectors_size = self.training_inputs.shape[0]
        random_indices = np.random.choice(training_feature_vectors_size, size=num_centers, replace=False)
        centers = self.training_inputs[random_indices]
        old_centers = centers.copy()
        closest_center_index = None
        converged = False

        while not converged:
            distances = cdist(centers, self.training_inputs)
            closest_center_index = np.argmin(distances, axis=0)
            centers = []
            for center_index in range(num_centers):
                data_points = self.training_inputs[closest_center_index == center_index]
                center = data_points.mean(axis=0) if data_points.shape[0] > 0 else data_points
                centers.append(center)
            centers = np.array(centers)
            converged = (old_centers - centers).sum() < 1e-6
            old_centers = centers.copy()

        return centers, closest_center_index

    def _calc_sigmas(self, centres, closest_center_index):
        sigmas = []
        for center_index, center in enumerate(centres):
            sigma = np.nanstd([euclidean(center, feature_vector) for feature_vector in
                               self.training_inputs[closest_center_index == center_index]])
            sigmas.append(sigma)
        # use mean sigma if center has less than 2 data points
        mean_sigma = np.nanmean(sigmas) * self.std_multiplier
        sigmas = [sigma * self.std_multiplier if sigma > 1e-6 else mean_sigma for sigma in sigmas]
        return sigmas

    def fit(self):
        phi = np.array(
            [self.gaussian_radial_basis_function(feature_vector, zip(self.centers, self.sigmas)) for feature_vector in
             self.training_inputs])

        lambda_identity = self.regularization_term * np.eye(phi.shape[1])

        return np.dot(np.dot(pinv(np.dot(phi.T, phi) + lambda_identity), phi.T), self.training_outputs)

    def predict(self, feature_vector):
        hidden = self.gaussian_radial_basis_function(feature_vector, zip(self.centers, self.sigmas))
        output = np.dot(hidden, self.weights)
        return output
