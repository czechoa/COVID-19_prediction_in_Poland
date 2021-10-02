import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

from LSTM.LSTM_one_sequences.data_processor import read_data_only_Poland, make_trainX_trainY, inverse_transform_train, \
    inverse_transform_test
from LSTM.LSTM_one_sequences.plot import Plot


def create_and_fit_model(_trainX, _trainY):
    features = _trainX.shape[2]
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, stateful=True, batch_input_shape=(1, None, features)))
    # model.add(LSTM(100, batch_input_shape=(1, None, features), stateful=True))
    model.add(LSTM(100, return_sequences=True, stateful=True))
    # model.add(LSTM(features,return_sequences=False,stateful=True))
    # model.add(Dense(features,activation='linear'))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mse", optimizer='adam')
    model.fit(_trainX, _trainY, epochs=5, batch_size=1, verbose=2)
    return model


def make_prediction_future(model, train, trainY, test):
    model.reset_states()

    train = train[np.newaxis, :, :]
    predictions_train = model.predict(train)

    future = []
    # currentStep = predictions[:, -1:, :]
    currentStep = trainY[:, -1:, :]
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

data_merge = read_data_only_Poland()
split = 0.83

train, test, trainX, trainY, _region, scaler = make_trainX_trainY(data_merge, split)

model = create_and_fit_model(trainX, trainY)
trainX = trainX[-1:, :, :]
trainY = trainY[-1:, :, :]

future_array, predictions_train = make_prediction_future(model, train, trainY, test)
trainPredict, trainY = inverse_transform_train(scaler, predictions_train, trainY)
testPredict, testY = inverse_transform_test(scaler, future_array, test)
plot = Plot(_region,trainPredict,testPredict,split)
plot.get_org_data_from_region_make_plot()

