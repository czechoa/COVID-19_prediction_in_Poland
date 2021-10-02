import math

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


# convert an array of values into a dataset matrix
def create_dataset(_dataset, _look_back=1):
    dataX, dataY = [], []
    for i in range(len(_dataset) - _look_back):
        a = _dataset[i:(i + _look_back), :]
        dataX.append(a)
        dataY.append(_dataset[i + _look_back, :])
    return np.array(dataX), np.array(dataY)


def load_dataset(_region, _columns_index=[-2, -1]):
    dataframe = _region.iloc[:, _columns_index]
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset


def normalise_dataset(_dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    _dataset = scaler.fit_transform(_dataset.reshape(len(_dataset), _dataset.shape[1]))
    return _dataset, scaler


def split_into_train_and_test_sets(dataset, split=0.85):
    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test


def reshape_train_to_one_sample(trainX, trainY):
    trainX = np.reshape(trainX, (1, trainX.shape[0], trainX.shape[2]))
    trainY = np.reshape(trainY, (1, trainY.shape[0], trainY.shape[1]))
    return trainX, trainY


def create_and_fit_model(trainX, trainY):
    features = trainX.shape[2]
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, stateful=True, batch_input_shape=(1, None, features)))
    # model.add(LSTM(100, batch_input_shape=(1, None, features), stateful=True))
    model.add(LSTM(100,return_sequences=True,stateful=True))
    # model.add(LSTM(features,return_sequences=False,stateful=True))
    # model.add(Dense(features,activation='linear'))
    model.add(Dense(1,activation="linear"))

    model.compile(loss="mse", optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    return model


def make_prediction_future(model, train,trainY, test):
    model.reset_states()

    train = train[np.newaxis,:, :]
    predictions_train = model.predict(train)

    future = []
    # currentStep = predictions[:, -1:, :]
    currentStep = trainY[:,-1:,:]
    # for i in range(testY.shape[0]):
    for i in range(test.shape[0]):

        currentStep = model.predict(currentStep)  # get the next step
        # currentStep = currentStep.reshape(1,1,)
        future.append(currentStep)  # store the future steps
    # after processing a sequence, reset the states for safety
    model.reset_states()

    future_array = np.array(future)
    future_array = future_array[:, 0, 0, :]

    features = train.shape[2]
    future_array = future_array.reshape(future_array.shape[0], features)
    return future_array, predictions_train


def reshape_back(train_f: np.array):
    train_f = train_f[0, :, :]
    # train_f = train_f.reshape(len(train_f),features)
    return train_f


def calculate_root_mean_squared_error(trainY, trainPredict, testY, testPredict):
    trainScore = math.sqrt(mean_squared_error(trainY[:, -1], trainPredict[:, -1]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:, -1], testPredict[:, -1]))
    print('Test Score: %.2f RMSE' % (testScore))


def inverse_transform_train(scaler, predictions, trainY):
    trainPredict = scaler.inverse_transform(reshape_back(predictions))
    trainY = scaler.inverse_transform(reshape_back(trainY))
    return trainPredict, trainY


def inverse_transform_test(scaler, future_array, testY):
    testPredict = scaler.inverse_transform(future_array)
    testY = scaler.inverse_transform(testY)
    return testPredict, testY


def shift_train_predictions_for_plotting(dataset, trainPredict, testPredict, look_back):
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    return trainPredictPlot, testPredictPlot


def plot_baseline_and_predictions(dataset, trainPredictPlot, testPredictPlot):
    for i in range(dataset.shape[1]):
        plt.plot(dataset[:, i],label= 'true data')
        plt.plot(trainPredictPlot[:, i], label= 'train prediction')
        plt.plot(testPredictPlot[:, i], label= 'prediction (future)')
        # plt.xlabel(xlabel="Date")
        # plt.ylabel(ylabel= )
        plt.title(i)
        plt.grid()
        plt.legend()
        plt.show()


def read_data_regions():
    data_merge = read_csv('data/data_lstm/data_all_with_one_hot_encode.csv')
    data_merge["region"].replace({"ŚŚ_average": "ŚŚŚ_average"}, inplace=True)
    data_merge = data_merge.sort_values(by=['region', 'date'])
    return data_merge

def read_data_only_Poland():
    data_merge = read_csv('data/data_lstm/merge_data_Poland.csv')
    data_merge["region"].replace({ "POLSKA":"ŚŚŚ_Poland"}, inplace=True)
    data_merge = data_merge.sort_values(by=['region', 'date'])
    return data_merge

def make_trainX_trainY():
    first = True
    x_full = np.array
    y_full = np.array
    columns_index = [-1]

    for region_name in data_merge['region'].unique()[0:]:
        if region_name == 'POLSKA':
            continue
        region = data_merge[data_merge['region'] == region_name]
        dataset = load_dataset(region, columns_index)
        dataset, scaler = normalise_dataset(dataset)

        train, test = split_into_train_and_test_sets(dataset,split=split)
        trainX, trainY = create_dataset(train)
        testX, testY = create_dataset(test)
        trainX, trainY = reshape_train_to_one_sample(trainX, trainY)

        if first:
            x_full = trainX
            y_full = trainY
            first = False
        else:
            x_full = np.concatenate((x_full, trainX), axis=0)
            y_full = np.concatenate((y_full, trainY), axis=0)

    return train, test, x_full,y_full, region, scaler


def get_org_data_from_region(region):
    date_train = region['date'].iloc[:int(region.shape[0] * split)]
    y_train_org = region['Engaged_respirator'].iloc[:int(region.shape[0] * split)]
    y_test_org = region['Engaged_respirator'].iloc[int(region.shape[0] * split):]
    date_test = region['date'].iloc[int(region.shape[0] * split):]
    return date_train, y_train_org, y_test_org, date_test


def get_org_data_from_region_make_plot(_region, _trainPredict, _testPredict):
    date_train, y_train_org, y_test_org, date_test = get_org_data_from_region(_region)
    trace1 = go.Scatter(
        x=date_train,
        y=y_train_org * 16,
        mode='lines',
        name='Data train'
    )
    trace2 = go.Scatter(
        # x=date_train[1:],
        x=date_train,
        y=_trainPredict[:, 0] * 16,
        mode='lines',
        name='Prediction train'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=_testPredict[:, 0] * 16,
        # y=testPredict*16,
        mode='lines',
        name='Prediction future'
    )

    trace5 = go.Scatter(
        x=date_test,
        y=y_test_org * 16,
        mode='lines',
        name='Ground true'
    )

    layout = go.Layout(
        title="Poland covid-19",
        xaxis={'title': "Date"},
        yaxis={'title': "Engaged respirator"}
    )

    fig = go.Figure(data=[trace1, trace2, trace3, trace5], layout=layout)

    fig.show()

# numpy.random.seed(7)

data_merge =read_data_only_Poland()

# look_back = 1
split = 0.83


train, test, trainX,trainY, _region, scaler = make_trainX_trainY()

model = create_and_fit_model(trainX, trainY)
trainX = trainX[-1:, :, :]
trainY = trainY[-1:, :, :]
# %%
future_array, predictions_train = make_prediction_future(model, train,trainY, test)
trainPredict, trainY = inverse_transform_train(scaler, predictions_train, trainY)
testPredict, testY = inverse_transform_test(scaler, future_array, test)
get_org_data_from_region_make_plot(_region, trainPredict, testPredict)



#
# # dataset = scaler.inverse_transform(dataset)
# trainPredictPlot, testPredictPlot = shift_train_predictions_for_plotting(dataset, trainPredict, testPredict, look_back)
# # plot_baseline_and_predictions(dataset, trainPredictPlot, testPredictPlot)
# # region = data_merge[data_merge['region'] == 'POLSKA']
#
# dataset = load_dataset(region,columns_index)
# plot_baseline_and_predictions(dataset*16, trainPredictPlot * 16, testPredictPlot * 16)

