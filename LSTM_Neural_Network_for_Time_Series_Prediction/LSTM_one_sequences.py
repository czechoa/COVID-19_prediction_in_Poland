import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from numpy import newaxis
import numpy as np
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def predict_sequence_full(self, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    return predicted

def predict_sequence_full_mine(self, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    return predicted

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
# data_merge = read_csv('LSTM_Neural_Network_for_Time_Series_Prediction/data/data_all_with_one_hot_encode.csv')
# dataframe = data_merge[data_merge['region'] == data_merge['region'].unique()[0]].iloc[:,-1]

data_merge = read_csv('LSTM_Neural_Network_for_Time_Series_Prediction/data/data_Poland_to_2021_05.csv')
dataframe = data_merge.iloc[:,-1]

# dataframe = read_csv('LSTM_Neural_Network_for_Time_Series_Prediction/data/region.csv',  engine='python')
# data = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',c)
# dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
# split into train and test sets
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, ( 1,trainX.shape[0], trainX.shape[1]))
trainY = numpy.reshape(trainY, ( 1,trainY.shape[0], 1))

# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, return_sequences=True,stateful=True,batch_input_shape=(1,None,1)))
# model.add(LSTM(70,return_sequences=True,stateful=True))
# model.add(LSTM(1,return_sequences=False,stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)


# model.reset_states()

# make predictions
# predictions = model.predict(trainX)
# testPredict = model.predict(testX)
# invert predictions
# predictions = model.predict(trainX)
future = []
# currentStep = predictions[:,-1:,:]
currentStep = trainX[:,-2:-1,:]
for i in range(testY.shape[0]):
    currentStep = model.predict(currentStep) #get the next step
    currentStep = currentStep.reshape(1,1,1)
    future.append(currentStep) #store the future steps
#after processing a sequence, reset the states for safety
model.reset_states()


future_array = np.array(future)
future_array = future_array[:,0,0,0]
future_array = future_array.reshape(future_array.shape[0],1)

# reshape back
def reshape_back(train_f :np.array):
    train_f = train_f[0,:,0]
    train_f = train_f.reshape(len(train_f),1)
    return  train_f

predictions = trainY
trainPredict = scaler.inverse_transform(reshape_back(predictions))
trainY = scaler.inverse_transform(reshape_back(trainY))
#
testPredict = scaler.inverse_transform(future_array)
testY = scaler.inverse_transform([testY])
#
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
