import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from radial_basis_function import RadialBasisFunctionNeuralNetwork
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


class BostonModel:
    def __init__(self, num_centers, regularization_term, std_multiplier):
        self.regularization_term = regularization_term
        self.std_multiplier = std_multiplier
        self.num_centers = num_centers
        self.features = ["RM", "LSTAT", "PTRATIO"]
        boston = load_boston()
        original_boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
        original_boston_df['Price'] = boston.target
        self.boston_df_inputs, self.boston_df_targets = self.remove_outliers(original_boston_df)
        self.pre_processed_boston_df = self.pre_process(self.boston_df_inputs)
        self.rbf = None
        self.train(num_centers, regularization_term, std_multiplier)
        #self.find_regularisation_term()
        #self.find_std_multiplier()
        # #self.find_optimal_center()

    def _get_feature_mean(self, feature):
        return self.boston_df_inputs[feature].mean()

    def _get_feature_std(self, feature):
        return self.boston_df_inputs[feature].std()

    # z-score normalisation
    def pre_process(self, inputs_df):
        boston_df = inputs_df[self.features]
        for feature in self.features:
            feature_mean = self._get_feature_mean(feature)
            feature_std = self._get_feature_std(feature)
            boston_df[feature] = boston_df[feature].apply(
                lambda val: (val - feature_mean) / feature_std)
        return boston_df

    @staticmethod
    def remove_outliers(original_boston_df):
        # # Remove rows with 50 value
        filtered_boston_df = original_boston_df[original_boston_df['Price'] != 50]
        return filtered_boston_df.drop('Price', axis=1), filtered_boston_df['Price']

    def train(self, num_centers, regularization_term, std_multiplier):
        training_inputs, testing_inputs, training_outputs, testing_outputs = train_test_split(
            self.pre_processed_boston_df.to_numpy(),
            self.boston_df_targets,
            test_size=0.2)

        self.rbf = RadialBasisFunctionNeuralNetwork(num_centers=num_centers, training_inputs=training_inputs,
                                                    training_outputs=training_outputs,
                                                    regularization_term=regularization_term,
                                                    std_multiplier=std_multiplier)

        training_predictions = np.array(
            [self.rbf.predict(feature_vec) for feature_vec in training_inputs])

        testing_predictions = np.array(
            [self.rbf.predict(feature_vec) for feature_vec in testing_inputs])

        validation_mse = mean_squared_error(testing_predictions, testing_outputs)
        training_mse = mean_squared_error(training_predictions, training_outputs)
        print("Training MSE is {}".format(training_mse))
        print("Validation MSE is {}".format(validation_mse))
        return training_mse, validation_mse

    def predict(self, inputs):
        return [self.rbf.predict(feature_vec) for feature_vec in inputs]

    def find_optimal_center(self):
        training_mse_list = []
        validation_mse_list = []
        for center_size in range(5, 350, 5):
            done = False
            while not done:
                try:
                    training_mse, validation_mse = self.train(center_size, self.regularization_term,
                                                              self.std_multiplier)
                    training_mse_list.append((center_size, training_mse))
                    validation_mse_list.append((center_size, validation_mse))
                    done = True
                except Exception:
                    continue
        plt.plot(*zip(*training_mse_list))
        plt.plot(*zip(*validation_mse_list))
        plt.title('Error Against increasing number of centers')
        plt.xlabel('Number of Centers')
        plt.ylabel('Holdout Error (MSE)')
        plt.legend(['Training Error', 'Validation Error'], loc='upper left')
        plt.show()
        print("Centers should be {}".format(min(validation_mse_list, key=lambda val: val[1])))

    def find_regularisation_term(self):
        training_mse_list = []
        validation_mse_list = []
        for regularization_term in np.arange(0.01, 2, 0.01):
            done = False
            while not done:
                try:
                    training_mse, validation_mse = self.train(self.num_centers, regularization_term,
                                                              self.std_multiplier)
                    training_mse_list.append((regularization_term, training_mse))
                    validation_mse_list.append((regularization_term, validation_mse))
                    done = True
                except Exception:
                    continue
        plt.plot(*zip(*training_mse_list))
        plt.plot(*zip(*validation_mse_list))
        plt.title('Error Against increasing regularization term')
        plt.xlabel('Regularization term')
        plt.ylabel('Holdout Error (MSE)')
        plt.legend(['Training Error', 'Validation Error'], loc='upper left')
        plt.show()
        print("Regularization term should be {}".format(min(validation_mse_list, key=lambda val: val[1])))

    def find_std_multiplier(self):
        training_mse_list = []
        validation_mse_list = []
        for std_multiplier in range(1, 20):
            done = False
            while not done:
                try:
                    training_mse, validation_mse = self.train(self.num_centers, self.regularization_term,
                                                              std_multiplier)
                    training_mse_list.append((std_multiplier, training_mse))
                    validation_mse_list.append((std_multiplier, validation_mse))
                    done = True
                except Exception:
                    continue
        plt.plot(*zip(*training_mse_list))
        plt.plot(*zip(*validation_mse_list))
        plt.title('Error Against increasing standard deviation multiplier')
        plt.xlabel('Standard Deviation Multiplier')
        plt.ylabel('Holdout Error (MSE)')
        plt.legend(['Training Error', 'Validation Error'], loc='upper left')
        plt.show()
        print("Standard Deviation Multiplier should be {}".format(min(validation_mse_list, key=lambda val: val[1])))
