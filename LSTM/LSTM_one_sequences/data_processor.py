import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler


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


def read_data_regions():
    data_merge = read_csv('data/data_lstm/data_all_with_one_hot_encode.csv')
    data_merge["region"].replace({"ŚŚ_average": "ŚŚŚ_average"}, inplace=True)
    data_merge = data_merge.sort_values(by=['region', 'date'])
    return data_merge


def read_data_only_Poland():
    data_merge = read_csv('data/data_lstm/merge_data_Poland.csv')
    data_merge["region"].replace({"POLSKA": "ŚŚŚ_Poland"}, inplace=True)
    data_merge = data_merge.sort_values(by=['region', 'date'])
    return data_merge


def inverse_transform_train(scaler, predictions, trainY):
    trainPredict = scaler.inverse_transform(reshape_back(predictions))
    trainY = scaler.inverse_transform(reshape_back(trainY))
    return trainPredict, trainY


def inverse_transform_test(scaler, future_array, testY):
    testPredict = scaler.inverse_transform(future_array)
    testY = scaler.inverse_transform(testY)
    return testPredict, testY


def reshape_back(train_f: np.array):
    train_f = train_f[0, :, :]
    return train_f


def make_trainX_trainY(data, split):
    first = True
    x_full = np.array
    y_full = np.array
    columns_index = [-1]

    for region_name in data['region'].unique()[0:]:
        if region_name == 'POLSKA':
            continue
        region = data[data['region'] == region_name]
        dataset = load_dataset(region, columns_index)
        dataset, scaler = normalise_dataset(dataset)

        train, test = split_into_train_and_test_sets(dataset, split=split)
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

    return train, test, x_full, y_full, region, scaler
