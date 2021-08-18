from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
import numpy as np
from tensorflow.python.keras.optimizer_v2.adam import Adam

np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_model(number_of_input_dim, layers_n):
    l2 = keras.regularizers.l2(l2=0.01)
    NN_model.add(layers.Dense(128, kernel_initializer='normal', input_dim=number_of_input_dim, activation='relu',
                              kernel_regularizer=l2))
    # The Hidden Layers :
    for i in range(layers_n):
        NN_model.add(layers.Dense(256, kernel_initializer='normal', activation='relu', kernel_regularizer=l2))

    # The Output Layer :
    NN_model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    NN_model.summary()


def train_model(train, target):
    checkpoint_name = 'outModel/Weights.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    NN_model.fit(train, target, epochs=30, batch_size=32, validation_split=0.2, callbacks=callbacks_list, verbose=0)


def compline_model():
    wights_file = 'outModel/Weights.hdf5'
    NN_model.load_weights(wights_file)  # load it
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


def make_submission_cvs(train: pd.DataFrame, target, sub_name):
    prediction = NN_model.predict(train)
    my_submission = pd.DataFrame({'region': train.index[:].to_list(),
                                  'date': target,
                                  'prediction': prediction[:, 0]})
    # my_submission.index[:] = train.index[:]
    my_submission.to_csv('{}.csv'.format(sub_name), index=False)
    print('A submission file has been made ', sub_name)


def make_submission(test: pd.DataFrame, days_ahead):
    prediction = NN_model.predict(test)
    name = 'prediction ' + str(days_ahead) + ' ahead'
    my_submission = pd.DataFrame(index=test.index, data={name: list(prediction[:, 0])})
    return my_submission

def make_future_submission(test: pd.DataFrame, day):
    prediction = NN_model.predict(test)
    my_submission = pd.DataFrame({'region': test.index.get_level_values(0),
                                  'date': day,
                                  'prediction': prediction[:, 0]})
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


def standard_scale(train, test, input_scaler=StandardScaler()):
    if input_scaler is not None:
        # fit scaler
        input_scaler.fit(train)
        # transform training dataset
        trainX: pd.DataFrame = pd.DataFrame(index=train.index[:], data=input_scaler.transform(train))

        # transform test dataset
        testX: pd.DataFrame = pd.DataFrame(index=test.index[:], data=input_scaler.transform(test))
    return trainX, testX


def make_all(train, target, layers_n=2):
    global NN_model
    NN_model = Sequential()
    make_model(train.shape[1], layers_n)
    train_model(train, target)
    compline_model()
