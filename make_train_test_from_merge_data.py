# %%
from merge_data_mobility_epidemic_situation import get_merge_data
import pandas as pd


# %%
def reshape_data_merge_to_get_train_with_two_week_history(data_merge: pd.DataFrame, number_of_days_in_one_rows):
    train_all = pd.DataFrame()
    n = number_of_days_in_one_rows
    number_of_days = len(data_merge.loc[:, 'date'].unique())
    while n < data_merge.shape[0]:
        if n % number_of_days == 0:  # new region
            n += number_of_days_in_one_rows
        else:
            data_merge_stack: pd.Series = data_merge.iloc[
                                          (
                                                  n - number_of_days_in_one_rows):n,
                                          3:].stack()
            data_merge_stack = pd.concat([data_merge.iloc[n, :3], data_merge_stack]).reset_index(
                drop=True)
            train_all = train_all.append(data_merge_stack, ignore_index=True)
            n += 1

    return train_all


# %%
def make_target(data_merge: pd.DataFrame, number_of_days_in_one_rows, number_day_ahead_to_predition):
    index = number_of_days_in_one_rows + number_day_ahead_to_predition
    # target = [data_merge.loc[data_merge['region'] == region].iloc[index:, -1] for region in
    #           data_merge.loc[:, 'region']]
    target = pd.Series()
    for region in data_merge.loc[:, 'region'].unique():
        target = target.append(data_merge.loc[data_merge['region'] == region].iloc[index:, -1])

    target = target.astype(float)
    return target


# %%
def make_test(data_merge: pd.DataFrame, number_of_days_in_one_rows, number_day_ahead_to_predition):
    test = pd.DataFrame()
    for region in data_merge.loc[:, 'region'].unique():
        data_merge_stack: pd.Series = data_merge.loc[data_merge['region'] == region].iloc[
                                      - number_of_days_in_one_rows - number_day_ahead_to_predition:-number_day_ahead_to_predition,
                                      3:].stack()
        last_day = data_merge.loc[data_merge['region'] == region].iloc[-1,:3]
        concat = pd.concat([last_day, data_merge_stack]).reset_index(drop=True)
        test = test.append(concat, ignore_index=True)
    test: pd.DataFrame = test.set_index([test.columns[0], test.columns[1]])
    test = test.astype(float)
    return test


# %%
def get_train_only_with_preparation_n_days_ahead(train_all: pd.DataFrame,
                                                 number_day_ahead_to_predition):
    train_f = pd.DataFrame()
    for region in train_all.iloc[:, 0].unique():
        one_region = train_all.loc[train_all[0] == region].iloc[:-number_day_ahead_to_predition, :]
        train_f = train_f.append(one_region)
    train_f: pd.DataFrame = train_f.set_index([train_all.columns[0], train_all.columns[1]])
    train_f = train_f.astype(float)
    return train_f


# %%
def get_train_test_target(data_merge: pd.DataFrame, train_all: pd.DataFrame,
                          number_of_days_in_one_rows, number_day_ahead_to_predition):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    test = make_test(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, test, target

# %%
from simple_regresion import *

period_of_time = 14
data_merge = get_merge_data()
train_all = reshape_data_merge_to_get_train_with_two_week_history(data_merge, period_of_time)


train, test, target = get_train_test_target(data_merge, train_all, period_of_time, 1)
make_all(train, target, 'results/prediction_7')
submission = make_submission(test, 1)
for i in range(2, 31):
    clear_model()
    train, test, target = get_train_test_target(data_merge, train_all, period_of_time, i)
    make_all(train, target, 'results/prediction_7')
    submission = add_prediction_to_submission(test, submission, i)
submission_to_cvs(submission, 'results/preparation_one_month_ahead_1')

