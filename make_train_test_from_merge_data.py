# %%
from merge_data_mobility_epidemic_situation import get_merge_data, get_merge_data_to_last_day
import pandas as pd


# %%
def reshape_data_merge_to_get_train_with_two_week_history(data_merge: pd.DataFrame, number_of_days_in_one_row,
                                                          first_n_attribute_dsc_region=4):
    train_all = pd.DataFrame()
    n = number_of_days_in_one_row
    number_of_days = len(data_merge.loc[:, 'date'].unique())
    while n < data_merge.shape[0]:
        if n % number_of_days == 0:  # new region
            n += number_of_days_in_one_row
        else:
            data_merge_stack: pd.Series = data_merge.iloc[
                                          (
                                                  n - number_of_days_in_one_row):n,
                                          first_n_attribute_dsc_region:].stack()
            data_merge_stack = pd.concat(
                [data_merge.iloc[n, :first_n_attribute_dsc_region], data_merge_stack]).reset_index(
                drop=True)
            train_all = train_all.append(data_merge_stack, ignore_index=True)
            n += 1

    return train_all


def reshape_data_merge_to_get_train_period_of_time_history(data_merge: pd.DataFrame, number_of_days_in_one_row,
                                                           first_n_attribute_dsc_region=4):
    train_all = pd.DataFrame()
    number_of_days = len(data_merge.loc[:, 'date'].unique())
    for region in data_merge.loc[:, 'region'].unique():
        region_df = data_merge.loc[data_merge['region'] == region]
        for n in range(number_of_days_in_one_row, number_of_days + 1):
            region_df_stack: pd.Series = region_df.iloc[
                                         (
                                                 n - number_of_days_in_one_row):n,
                                         first_n_attribute_dsc_region:].stack()
            region_df_stack = pd.concat(
                [region_df.iloc[n - 1, :first_n_attribute_dsc_region], region_df_stack]).reset_index(
                drop=True)
            train_all = train_all.append(region_df_stack, ignore_index=True)
            n += 1

    return train_all


def make_target(data_merge: pd.DataFrame, number_of_days_in_one_rows, number_day_ahead_to_predition):
    # index = number_of_days_in_one_rows + number_day_ahead_to_predition 
    index = number_of_days_in_one_rows + number_day_ahead_to_predition - 1

    # target = [data_merge.loc[data_merge['region'] == region].iloc[index:, -1] for region in
    #           data_merge.loc[:, 'region']]
    target_f = pd.Series()
    for region in data_merge.loc[:, 'region'].unique():
        target_f = target_f.append(data_merge.loc[data_merge['region'] == region].iloc[index:, -1])

    target_f = target_f.astype(float)
    return target_f


def make_test(data_merge: pd.DataFrame, number_of_days_in_one_rows, number_day_ahead_to_predition,
              first_n_attribute_dsc_region):
    test_f = pd.DataFrame()
    for region in data_merge.loc[:, 'region'].unique():
        data_merge_stack: pd.Series = data_merge.loc[data_merge['region'] == region].iloc[

                                      - number_of_days_in_one_rows - number_day_ahead_to_predition:-number_day_ahead_to_predition,
                                      first_n_attribute_dsc_region:].stack()
        last_day = data_merge.loc[data_merge['region'] == region].iloc[-1, :first_n_attribute_dsc_region]
        concat = pd.concat([last_day, data_merge_stack]).reset_index(drop=True)
        test_f = test_f.append(concat, ignore_index=True)
    test_f: pd.DataFrame = test_f.set_index([test_f.columns[0], test_f.columns[1]])
    test_f = test_f.astype(float)
    return test_f


def make_date_to_prediction(train_all: pd.DataFrame):
    date_to_pr = pd.DataFrame()
    for region in train_all.loc[:, 0].unique():
        region_df = train_all.loc[train_all[0] == region]
        date_to_pr = date_to_pr.append(region_df.iloc[-1, :])
    date_to_pr: pd.DataFrame = date_to_pr.set_index([date_to_pr.columns[0], date_to_pr.columns[1]])
    date_to_pr = date_to_pr.astype(float)
    return date_to_pr


def get_train_only_with_preparation_n_days_ahead(train_all: pd.DataFrame,
                                                 number_day_ahead_to_predition):
    train_f = pd.DataFrame()
    for region in train_all.iloc[:, 0].unique():
        one_region = train_all.loc[train_all[0] == region].iloc[:-number_day_ahead_to_predition, :]
        train_f = train_f.append(one_region)
    train_f: pd.DataFrame = train_f.set_index([train_all.columns[0], train_all.columns[1]])
    train_f = train_f.astype(float)
    return train_f


def get_train_test_target(data_merge: pd.DataFrame, train_all: pd.DataFrame,
                          number_of_days_in_one_rows, number_day_ahead_to_predition, first_n_attribute_dsc_region=4):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    test = make_test(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition,
                     first_n_attribute_dsc_region)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, test, target


def get_train_target(data_merge: pd.DataFrame, train_all: pd.DataFrame,
                     number_of_days_in_one_rows, number_day_ahead_to_predition, first_n_attribute_dsc_region=4):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, target


# %%

# from simple_regresion import *
# #
# def one_mounTH_prediction():
#     period_of_time = 14
#     data_merge = get_merge_data()
#     train_all = reshape_data_merge_to_get_train_with_two_week_history(data_merge, period_of_time)
#     train, test, target = get_train_test_target(data_merge, train_all, period_of_time, 1)
#     make_all(train, target, 'results/prediction_7')
#     submission = make_submission(test, 1)
#     for i in range(2, 31):
#         clear_model()
#         train, test, target = get_train_test_target(data_merge, train_all, period_of_time, i)
#         make_all(train, target, 'results/prediction_7')
#         submission = add_prediction_to_submission(test, submission, i)
#     submission_to_cvs(submission, 'results/preparation_one_month_ahead_1')
#

def get_all_train_test_target(period_of_time=14, day_ahead=7
                              , first_n_attribute_dsc_region=4, last_day_train="2021-04-01"):
    # data_merge = get_merge_data()
    data_merge = get_merge_data_to_last_day(last_day_train)
    train_all = reshape_data_merge_to_get_train_with_two_week_history(data_merge, period_of_time,
                                                                      first_n_attribute_dsc_region)
    train, test, target = get_train_test_target(data_merge, train_all, period_of_time, day_ahead,
                                                first_n_attribute_dsc_region)
    return train, test, target

# %%

# train, test, target = get_all_train_test_target()
