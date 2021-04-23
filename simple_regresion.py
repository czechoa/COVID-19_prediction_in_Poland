from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

NN_model = Sequential()


# %%
def make_model(numberOfInput_dim):
    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=numberOfInput_dim, activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()


# %%
def train_model(train, target):
    checkpoint_name = 'outModel/Weights.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    NN_model.fit(train, target, epochs=500, batch_size=32, validation_split=0.2, callbacks=callbacks_list, verbose=0)


# %%
def compline_model():
    wights_file = 'outModel/Weights.hdf5'
    NN_model.load_weights(wights_file)  # load it
    print(wights_file)
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# %%
def make_submission_cvs( train:pd.DataFrame, target, sub_name):
    prediction = NN_model.predict(train)
    my_submission = pd.DataFrame({'Id': train.index[:].to_list()
                                     , 'Target': target,
                                  'Prediction': prediction[:, 0]})
    # my_submission.index[:] = train.index[:]
    my_submission.to_csv('{}.csv'.format(sub_name), index=False)
    print('A submission file has been made ',sub_name)


# %%
def make_all(train, target, sub_name):
    make_model(train.shape[1])
    train_model(train, target)
    compline_model()
    make_submission_cvs(train, target, sub_name)
