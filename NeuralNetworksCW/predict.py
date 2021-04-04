import pandas as pd
from sklearn.datasets import load_boston
from model import BostonModel

# Hyperparams
NUM_CENTERS = 60
LAMBDA = 0.38
STD_MULTIPLIER = 15

boston_feature_names = list(load_boston().feature_names)
prediction_inputs_df = pd.read_csv('prediction_inputs', names=boston_feature_names, delimiter=' ')

boston_rbf_model = BostonModel(NUM_CENTERS, LAMBDA, STD_MULTIPLIER)
pre_processed_prediction_inputs_df = boston_rbf_model.pre_process(prediction_inputs_df)
pre_processed_prediction_inputs = pre_processed_prediction_inputs_df.to_numpy()

predictions = boston_rbf_model.predict(pre_processed_prediction_inputs)
formatted_predictions = [str(val) for val in predictions]

with open('predictions.txt', 'w') as text_file:
    for val in formatted_predictions:
        text_file.write(val + '\n')

print('Finished making predictions!')
