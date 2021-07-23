__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from LSTM.modificate_jaungiers_LSTM.core.data_processor import DataLoader
from LSTM.modificate_jaungiers_LSTM.core.model import Model


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


def normalised_data(path='data/data_lstm/data_Poland_to_2021_05.csv'):
    df = pd.read_csv(path)
    data_df = df.iloc[:, -2:]
    data_desc = data_df.describe()
    data_org = data_df.values

    data_nor = (data_org - data_desc.loc['mean'].values) / (data_desc.loc['max'].values - data_desc.loc['min'].values)
    # data_nor = (data_org) / (data_desc.loc['max'].values - data_desc.loc['min'].values)

    df.iloc[:, -2:] = data_nor
    # df_ns = pd.DataFrame(columns=data_df.columns, data=data_nor)
    # df_ns.insert(0,'region',df['region'])
    df.to_csv(path[:-4] + '_ns.csv', index=False)
    return data_desc


def normalised_data_min_max(path='data/data_lstm/data_Poland_to_2021_05.csv'):
    df = pd.read_csv(path)
    data_df = df.iloc[:, 3:]
    data_desc = data_df.describe()
    data_org = data_df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_nor = scaler.fit_transform(data_org)

    df.iloc[:, 3:] = data_nor
    df.to_csv(path[:-4] + '_ns.csv', index=False)
    return data_desc


# data_desc = normalised_data_min_max('LSTM/data/data_all_with_one_hot_encode.csv')
data_merge = pd.read_csv('data/data_lstm/data_all_with_one_hot_encode.csv')
first = True
for region in data_merge.loc[:, 'region'].unique()[::-1]:

    if region == 'POLSKA':
        continue

    data_region = data_merge[data_merge['region'] == region]
    data_region.to_csv('data/data_lstm/region.csv')
    data_desc = normalised_data_min_max('data/data_lstm/region.csv')

    configs = json.load(open('LSTM/modificate_jaungiers_LSTM/config.json', 'r'))
    data = DataLoader(
        os.path.join('data/data_lstm', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['sequence_length']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    if first:
        x_full = x
        y_full = y
        first = False


    else:
        x_full = np.concatenate((x_full, x), axis=0)
        y_full = np.concatenate((y_full, y), axis=0)
    # x_full = np.insert(x_full, x)
    # y_full = np.insert(y_full, y)
    break
x = x_full
y = y_full
x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)
model.train(
    x,
    y,
    epochs=configs['training']['epochs'],
    batch_size=configs['training']['batch_size']
    , save_dir=configs['model']['save_dir']
)
# %%
# predictions = model.predict_point_by_point(x_test)

# predictions = np.array(predictions) * (data_desc.loc['max'][-1] - data_desc.loc['min'][-1]) + data_desc.loc['mean'][-1]
# y_test_sc = np.array(y_test) * (data_desc.loc['max'][-1] - data_desc.loc['min'][-1]) + data_desc.loc['mean'][-1]

# predictions = predictions * (data_org.max() - data_org.min()) + data_org.mean()
# y_test = y_test * (data_org.max() - data_org.min()) + data_org.mean()

# plot_results(predictions, y_test)

predictions_full: list = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
# predictions_sc = np.array(predictions_full) * (data_desc.loc['max'][-1] - data_desc.loc['min'][-1]) + \
#                  data_desc.loc['mean'][-1]

plot_results(predictions_full, y_test)

# plot_results(predictions_sc, y_test_sc)
# %%
# b = data_merge.columns().unique()