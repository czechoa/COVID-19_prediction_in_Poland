from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
import numpy as np

np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# %%
def make_model(numberOfInput_dim, layers_n):
    # The Input Layer :
    # NN_model.add(norm)
    print(numberOfInput_dim)
    # NN_model.add(layers.Dense(128, kernel_initializer='normal', input_dim=numberOfInput_dim, activation='relu'))
    # print(numberOfInput_dim)
    NN_model.add(layers.Dense(numberOfInput_dim, kernel_initializer='normal',input_dim=numberOfInput_dim,  activation='relu'))
    # The Hidden Layers :
    # two hidden layer because repression all not function
    for i in range(layers_n):
        NN_model.add(layers.Dense(512, kernel_initializer='normal', activation='relu'))
    # NN_model.add(Dropout(0.2))
    # The Output Layer :
    NN_model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()


def make_model_from_audio( numberOfInput_dim,layers_n):
    NN_model.add(layers.Dense(128, kernel_initializer='normal', input_dim=numberOfInput_dim, activation='relu'))
    NN_model.add(layers.Dense(64, activation='swish'))
    NN_model.add(layers.Dense(64, activation='relu'))
    NN_model.add(layers.Dense(64, activation='swish'))
    NN_model.add(layers.Dense(64, activation='relu'))
    NN_model.add(layers.Dense(64, activation='swish'))
    NN_model.add(layers.Dense(1))
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()


def train_model(train, target):
    checkpoint_name = 'outModel/Weights.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    NN_model.fit(train, target, epochs=15, batch_size=32, validation_split=0.2, callbacks=callbacks_list, verbose=0)


def compline_model():
    wights_file = 'outModel/Weights.hdf5'
    NN_model.load_weights(wights_file)  # load it
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


def make_submission_cvs(train: pd.DataFrame, target, sub_name):
    prediction = NN_model.predict(train)
    my_submission = pd.DataFrame({'Id': train.index[:].to_list()
                                     , 'Target': target,
                                  'Prediction': prediction[:, 0]})
    # my_submission.index[:] = train.index[:]
    my_submission.to_csv('{}.csv'.format(sub_name), index=False)
    print('A submission file has been made ', sub_name)


def make_submission(test: pd.DataFrame, days_ahead):
    prediction = NN_model.predict(test)
    name = 'prediction ' + str(days_ahead) + ' ahead'
    my_submission = pd.DataFrame(index=test.index, data={name: list(prediction[:, 0])})
    return my_submission


def add_prediction_to_submission(test: pd.DataFrame, my_submission: pd.DataFrame, day_ahead):
    prediction = NN_model.predict(test)
    my_submission['predition ' + str(day_ahead) + ' ahead'] = prediction[:, 0]
    return my_submission


def submission_to_cvs(my_submission: pd.DataFrame, sub_name):
    my_submission.to_csv('{}.csv'.format(sub_name), index=False)
    print('A submission file has been made ', sub_name)


def clear_model():
    clear_session()


def standardScaler(train, test, input_scaler=StandardScaler()):
    if input_scaler is not None:
        # fit scaler
        input_scaler.fit(train)
        # transform training dataset
        trainX: pd.DataFrame = pd.DataFrame(index=train.index[:], data=input_scaler.transform(train))

        # transform test dataset
        testX: pd.DataFrame = pd.DataFrame(index=test.index[:], data=input_scaler.transform(test))
    return trainX, testX


def make_all(train, target, layers_n = 2):
    global NN_model
    NN_model = Sequential()
    # norm = preprocessing.Normalization()
    # norm.adapt(np.array(train))
    print('start')
    make_model(train.shape[1], layers_n)
    train_model(train, target)
    compline_model()
    # make_submission_cvs(train, target, sub_name, )


# %%
from make_train_test_from_merge_data import get_all_train_test_target

train, test, target = get_all_train_test_target(period_of_time=21)
# %%
train_sc, test_sc = standardScaler(train, test, input_scaler=MinMaxScaler())
# %%
# import time
from prepare_data_epidemic_situation_in_regions import *

layer_err_mean = list()
for layers_n in range(2,3):
    layer_err_sum = 0
    for i in range(5):
        # start_time = time.time()
        make_all(train_sc, target, layers_n)
        submission = make_submission(test_sc, 7)
        # print("--- %s seconds ---" % (time.time() - start_time))
        clear_model()

        submission = submission.reset_index()
        test_ahead: pd.DataFrame = get_test_respiration()
        submission.rename(columns={submission.columns[0]: test_ahead.columns[1], submission.columns[2]: 'prediction'},
                          inplace=True)

        result = pd.merge(test_ahead, submission.drop(columns=submission.columns[1]), on=test_ahead.columns[1])
        result_err = result.iloc[:, :2]
        result_err['subtract'] = result.iloc[:, -2].astype(float) - result.iloc[:, -1].astype(float)
        result_err['relative error in %'] = abs(result_err.loc[:, 'subtract']/result.iloc[:, -1].astype(float))*100
        norm_2 = np.linalg.norm(result_err['relative error in %'], ord=2)
        layer_err_sum = layer_err_sum + norm_2
    layer_err_mean.append( layer_err_sum/10)
print(layer_err_mean)
# %%
hiden_layers_result = pd.DataFrame({'layer':range(5),'mean_norm_2_result':layer_err_mean})
