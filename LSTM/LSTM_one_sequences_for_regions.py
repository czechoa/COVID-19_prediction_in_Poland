import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


def load_dataset(region, columns_index=[-2, -1]):
    #  fix random seed for reproducibility
    # load the dataset
    # data_merge = read_csv('LSTM/data/data_all_with_one_hot_encode.csv')
    # dataframe = data_merge[data_merge['region'] == data_merge['region'].unique()[0]].iloc[:,-1]

    # data_merge = read_csv('LSTM/data/data_Poland_to_2021_05.csv')
    dataframe = region.iloc[:, columns_index]

    # dataframe = read_csv('LSTM/data/region.csv',  engine='python')
    # data = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',c)
    # dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset


def normalise_dataset(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(len(dataset), dataset.shape[1]))
    return dataset, scaler


def split_into_train_and_test_sets(split=0.85):
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
    # model.add(LSTM(4, batch_input_shape=(1, None, features), stateful=True))
    # model.add(LSTM(5,return_sequences=True,stateful=True))
    # model.add(LSTM(features,return_sequences=False,stateful=True))
    # model.add(Dense(features,activation='linear'))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    return model


def make_prediction_future(model, trainX, testY):
    model.reset_states()
    # make predictions
    predictions = model.predict(trainX)
    # testPredict = model.predict(testX)
    # invert predictions
    # predictions = model.predict(trainX)
    future = []
    currentStep = predictions[:, -1:, :]
    currentStep = trainY[:,-1:,:]
    # for i in range(testY.shape[0]):
    for i in range(testY.shape[0]):

        currentStep = model.predict(currentStep)  # get the next step
        # currentStep = currentStep.reshape(1,1,)
        future.append(currentStep)  # store the future steps
    # after processing a sequence, reset the states for safety
    model.reset_states()

    future_array = np.array(future)
    future_array = future_array[:, 0, 0, :]

    features = trainX.shape[2]
    future_array = future_array.reshape(future_array.shape[0], features)
    return future_array, predictions


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

# numpy.random.seed(7)

look_back = 1
split = 0.78
data_merge = read_csv('data/data_lstm/data_all_with_one_hot_encode.csv')
data_merge["region"].replace({"ŚŚ_average":"ŚŚŚ_average"}, inplace=True)
data_merge = data_merge.sort_values(by=['region', 'date'])
columns_index=[-1]
# columns_index= range(data_merge.shape[1]-1,data_merge.shape[1])

first = True
x_full = np.array
y_full = np.array

for region_name in data_merge['region'].unique()[0:]:
    if region_name == 'POLSKA':
        continue
    region = data_merge[data_merge['region'] == region_name]
    dataset = load_dataset(region, columns_index)
    dataset, scaler = normalise_dataset(dataset)
    train, test = split_into_train_and_test_sets(split=split)

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

trainX = x_full
trainY = y_full

model = create_and_fit_model(trainX, trainY)
trainX = trainX[-1:, :, :]
trainY = trainY[-1:, :, :]

# trainX = trainX.reshape(1,len(trainX),trainX.shape[1])
future_array, predictions_train = make_prediction_future(model, trainX, testY)

trainPredict, trainY = inverse_transform_train(scaler, predictions_train, trainY)
testPredict, testY = inverse_transform_test(scaler, future_array, testY)

dataset = scaler.inverse_transform(dataset)

trainPredictPlot, testPredictPlot = shift_train_predictions_for_plotting(dataset, trainPredict, testPredict, look_back)
# plot_baseline_and_predictions(dataset, trainPredictPlot, testPredictPlot)

region = data_merge[data_merge['region'] == 'POLSKA']

dataset = load_dataset(region,columns_index)
plot_baseline_and_predictions(dataset, trainPredictPlot * 16, testPredictPlot * 16)
