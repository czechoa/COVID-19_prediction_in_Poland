# %%
import pandas as pd
import numpy as np

# first_n_attribute_dsc_region= 5
first_n_attribute_dsc_region = 3 + 17  # region date and day of the week plus oneHotCode (region)


def avarage_train_from_n_days(train_f: pd.DataFrame, days_n):
    # TODO  from  train_all
    # iterate over each group
    df1_grouped = train_f.groupby(train_f.columns[0])
    train_mean_n_days = pd.DataFrame()
    for group_name, df_group in df1_grouped:
        df_region = df_group.rolling(days_n).mean()
        train_mean_n_days = train_mean_n_days.append(df_region)
    train_mean_n_days = train_mean_n_days.dropna()

    return train_mean_n_days


def reshape_data_merge_to_get_train_period_of_time_history(data_merge: pd.DataFrame, number_of_days_in_one_row):
    train_all = pd.DataFrame()

    data_merge = oneHotEncode(data_merge, 'region')

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
    first_index = number_of_days_in_one_rows + number_day_ahead_to_predition - 1

    # target = [data_merge.loc[data_merge['region'] == region].iloc[index:, -1] for region in
    #           data_merge.loc[:, 'region']]
    target_f = pd.DataFrame()
    for region in data_merge.loc[:, 'region'].unique():
        one_region_target: pd.DataFrame = data_merge[data_merge['region'] == region].iloc[first_index:, [0, 1, -1]]
        one_region_target = one_region_target.set_index([one_region_target.columns[0], one_region_target.columns[1]])
        target_f = target_f.append(one_region_target)

    target_f = target_f.astype(float)
    return target_f


def make_test(data_merge: pd.DataFrame, number_of_days_in_one_rows, number_day_ahead_to_predition):
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
                          number_of_days_in_one_rows, number_day_ahead_to_predition):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    test = make_test(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition,
                     first_n_attribute_dsc_region)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, test, target


def get_train_test_target(data_merge: pd.DataFrame, train_all: pd.DataFrame,
                          number_of_days_in_one_rows, number_day_ahead_to_predition):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    test = make_test(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition,
                     first_n_attribute_dsc_region)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, test, target


def get_train_target(data_merge: pd.DataFrame, train_all: pd.DataFrame,
                     number_of_days_in_one_rows, number_day_ahead_to_predition):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, target


def oneHotEncode(df, colName):
    if df[colName].dtype == np.dtype('object'):
        dummies = pd.get_dummies(df[colName], prefix=colName)
        col_dsc = first_n_attribute_dsc_region - len(df['region'].unique())
        df = pd.concat([df.iloc[:, :col_dsc], dummies, df.iloc[:, col_dsc:]], axis=1)

        # drop the encoded column
        # df.drop([colName], axis=1, inplace=True)
    return df

# def get_all_train_test_target(period_of_time=14, day_ahead=7
#                               , first_n_attribute_dsc_region=4, last_day_train="2021-04-01"):
#     # data_merge = get_merge_data()
#     data_merge = get_merge_data_to_last_day(last_day_train)
#     train_all = reshape_data_merge_to_get_train_with_two_week_history(data_merge, period_of_time,
#                                                                       first_n_attribute_dsc_region)
#     train, test, target = get_train_test_target(data_merge, train_all, period_of_time, day_ahead,
#                                                 first_n_attribute_dsc_region)
#     return train, test, target

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


# %%

# train, test, target = get_all_train_test_target()
