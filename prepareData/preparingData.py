import pandas as pd
import numpy as np
from simple_regresion import *


# %%
def get_data():
    # get train data
    train_data_path = '../data/train.csv'
    train = pd.read_csv(train_data_path)

    # get test data
    test_data_path = '../data/test.csv'
    test = pd.read_csv(test_data_path)

    return train, test


def get_combined_data():
    # reading train data
    train, test = get_data()

    target = train.SalePrice
    train.drop(['SalePrice'], axis=1, inplace=True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Id'], inplace=True, axis=1)
    return combined, target


# Load train and test data into pandas DataFrames
train_data, test_data = get_data()

# Combine train and test data to process them together
combined, target = get_combined_data()


# %%

def get_cols_with_no_nans(df, col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else:
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')
print('Number of numerical columns with no nan values :', len(num_cols))
print('Number of nun-numerical columns with no nan values :', len(cat_cols))
print(combined.shape)
combined = combined[num_cols + cat_cols]


# %%
def oneHotEncode(df, colNames):
    for col in colNames:
        if (df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


combined = oneHotEncode(combined, cat_cols)


def split_combined(combined):
    train = combined[:1460]
    test = combined[1460:]

    return train, test


train, test = split_combined(combined)
print(train)
# %%
# make_all(train, target,"result")

make_submission_cvs(train, target, 'data/result')
# model.make_submission_cvs(train, target, 'data/result')
