__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from LSTM.modificate_jaungiers_LSTM.core.data_processor import DataLoader
from LSTM.modificate_jaungiers_LSTM.core.model import Model
import plotly.graph_objects as go


def normalised_data_min_max(df):
    data_df = df.iloc[:, 3:]
    data_org = data_df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_nor = scaler.fit_transform(data_org)

    df.iloc[:, 3:] = data_nor
    return df


def read_data_and_reorder(path = "data/data_lstm/data_all_with_one_hot_encode.csv"):
    data_merge = pd.read_csv(path)
    data_merge["region"].replace({"ŚŚ_average": "ŚŚŚ_average"}, inplace=True)
    data_merge_org = data_merge.sort_values(by=['region', 'date'])
    data_merge = data_merge_org[data_merge['region'] != 'POLSKA']
    return data_merge

data_merge = read_data_and_reorder()
data_merge = normalised_data_min_max(data_merge)
first = True
for region in data_merge.loc[:, 'region'].unique():

    data_region = data_merge[data_merge['region'] == region]
    data_region.to_csv('data/data_lstm/region.csv', index=False)

    configs = json.load(open('LSTM/modificate_jaungiers_LSTM/config.json', 'r'))
    data = DataLoader(
        os.path.join('data/data_lstm', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['sequence_length']
    )


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

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)
model = Model()
model.build_model(configs)
model.train(
    x_full,
    y_full,
    epochs=configs['training']['epochs'],
    batch_size=configs['training']['batch_size']
    , save_dir=configs['model']['save_dir']
)

predictions_full: list = model.predict_sequence_full(x_test, configs['data']['sequence_length'])

date_train = data_region['date'].iloc[:int(data_region.shape[0] * configs['data']['train_test_split'])]
date_test = data_region['date'].iloc[int(data_region.shape[0] * configs['data']['train_test_split']):]

y_train_org = data_region['Engaged_respirator'].iloc[:int(data_region.shape[0] * configs['data']['train_test_split'])]
y_test_org = data_region['Engaged_respirator'].iloc[int(data_region.shape[0] * configs['data']['train_test_split']):]

predictions_train = model.predict_point_by_point(x)


trace2 = go.Scatter(
    x=date_train,
    y=y[:, 0],
    mode='lines',
    name='train'
)
trace3 = go.Scatter(
    x=date_test,
    y=y_test[:, 0],
    mode='lines',
    name='ground Truth'
)
trace4 = go.Scatter(
    x=date_test,
    y=predictions_full,
    mode='lines',
    name='future prediction'
)

trace6 = go.Scatter(
    x=date_train,
    y=predictions_train,
    mode='lines',
    name='train prediction'
)
layout = go.Layout(
    title="Google Stock",
    xaxis={'title': "Date"},
    yaxis={'title': "Close"}
)

fig = go.Figure(data=[trace2,trace3, trace4,trace6], layout=layout)

fig.show()
# %%
model.predict_single(x_test,configs['data']['sequence_length'])


