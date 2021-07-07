__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
from LSTM_Neural_Network_for_Time_Series_Prediction.core.data_processor import DataLoader
from LSTM_Neural_Network_for_Time_Series_Prediction.core.model import Model
import numpy as np
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

df = pd.read_csv('LSTM_Neural_Network_for_Time_Series_Prediction/data/data_Poland_to_2021_05.csv')
data_df = df.iloc[:,[-1]]
data_desc = data_df.describe()
data = data_df.values

data_nor = (data - data_desc.loc['mean'].values)/ (data_desc.loc['max'].values - data_desc.loc['min'].values)
# data_nor = (data - data_desc.loc['mean'])/ (data_desc.loc['max'] - data_desc.loc['min'])

df_ns = pd.DataFrame(columns= data_df.columns, data = data_nor)

df_ns.to_csv('LSTM_Neural_Network_for_Time_Series_Prediction/data/data_Poland_to_2021_05_ns.csv', index= False)

configs = json.load(open('LSTM_Neural_Network_for_Time_Series_Prediction/config.json', 'r'))

data = DataLoader(
	os.path.join('LSTM_Neural_Network_for_Time_Series_Prediction/data', configs['data']['filename']),
	configs['data']['train_test_split'],
	configs['data']['columns']
)


model = Model()
model.build_model(configs)
x, y = data.get_train_data(
	seq_len = configs['data']['sequence_length'],
	normalise = configs['data']['normalise']
)
model.train(
	x,
	y,
	epochs = configs['training']['epochs'],
	batch_size = configs['training']['batch_size']
    ,save_dir= configs['model']['save_dir']
)
x_test, y_test = data.get_test_data(
	seq_len = configs['data']['sequence_length'],
	normalise = configs['data']['normalise']
)

predictions = model.predict_point_by_point(x_test)
predictions = np.array( predictions) * (data_desc.loc['max'][-1] -  data_desc.loc['min'][-1] ) + data_desc.loc['mean'][-1]

# predictions = predictions * (data_org.max() - data_org.min()) + data_org.mean()
# y_test = y_test * (data_org.max() - data_org.min()) + data_org.mean()
# # data.denormalise_windows(y_test)
# # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
predictions_full:list = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
predictions_sc = np.array( predictions_full) * (data_desc.loc['max'][-1] -  data_desc.loc['min'][-1] ) + data_desc.loc['mean'][-1]


y_test_sc = np.array( y_test) * (data_desc.loc['max'][-1] -  data_desc.loc['min'][-1] ) + data_desc.loc['mean'][-1]


plot_results(predictions_sc,y_test_sc)
plot_results(predictions,y_test_sc)


