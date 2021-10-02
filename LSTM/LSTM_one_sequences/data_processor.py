import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Data_processor:
    def __init__(self, data,split):
        self.split = split
        self.data = data
        self.x_full = np.array
        self.y_full = np.array
        self.columns_index = [-1]

    def create_dataset(self, _dataset, _look_back=1):
        dataX, dataY = [], []
        for i in range(len(_dataset) - _look_back):
            a = _dataset[i:(i + _look_back), :]
            dataX.append(a)
            dataY.append(_dataset[i + _look_back, :])
        return np.array(dataX), np.array(dataY)

    def load_dataset(self, region):
        dataframe = region.iloc[:,self.columns_index]
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataset


    def normalise_dataset(self, dataset):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset.reshape(len(dataset), dataset.shape[1]))
        return dataset, scaler


    def split_into_train_and_test_sets(self, dataset):
        train_size = int(len(dataset) * self.split)
        # test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        return train, test


    def reshape_train_to_one_sample(self,trainX, trainY):
        trainX = np.reshape(trainX, (1, trainX.shape[0], trainX.shape[2]))
        trainY = np.reshape(trainY, (1, trainY.shape[0], trainY.shape[1]))
        return trainX, trainY

    def inverse_transform_train(self, scaler, predictions, trainY):
        trainPredict = scaler.inverse_transform(self.reshape_back(predictions))
        trainY = scaler.inverse_transform(self.reshape_back(trainY))
        return trainPredict, trainY


    def inverse_transform_test(self, scaler, future_array, testY):
        testPredict = scaler.inverse_transform(future_array)
        testY = scaler.inverse_transform(testY)
        return testPredict, testY


    def reshape_back(self,train_f: np.array):
        train_f = train_f[0, :, :]
        return train_f


    def make_trainX_trainY(self):
        global x_full, y_full
        first = True

        for region_name in self.data['region'].unique()[0:]:
            if region_name == 'POLSKA':
                continue
            region = self.data[self.data['region'] == region_name]
            dataset = self.load_dataset(region)
            dataset, scaler = self.normalise_dataset(dataset)

            train, test = self.split_into_train_and_test_sets(dataset)
            trainX, trainY = self.create_dataset(train)
            # testX, testY = create_dataset(test)
            trainX, trainY = self.reshape_train_to_one_sample(trainX, trainY)

            if first:
                x_full = trainX
                y_full = trainY
                first = False
            else:
                x_full = np.concatenate((x_full, trainX), axis=0)
                y_full = np.concatenate((y_full, trainY), axis=0)

        return train, test, x_full, y_full, region, scaler
