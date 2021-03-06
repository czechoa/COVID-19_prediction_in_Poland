import pandas as pd
import numpy as np


def averaged_merge_data_from_n_days(merge_data: pd.DataFrame, days_n, first_n_attribute_dsc_region=5):
    df1_grouped = merge_data.groupby(merge_data.columns[0])
    train_mean_n_days = pd.DataFrame()
    for group_name, df_group in df1_grouped:
        df_region = df_group.iloc[:, first_n_attribute_dsc_region:].rolling(days_n).mean()
        train_mean_n_days = train_mean_n_days.append(df_region)

    merge_data.iloc[:, first_n_attribute_dsc_region:] = train_mean_n_days
    merge_data = merge_data.dropna()
    return merge_data


def reshape_data_merge_to_get_train_period_of_time_history(data_merge: pd.DataFrame, number_of_days_in_one_row,
                                                           first_n_attribute_dsc_region=5):
    train_all = pd.DataFrame()

    data_merge, first_n_attribute_dsc_region = one_hot_encode(data_merge, 'region', first_n_attribute_dsc_region)

    number_of_days = len(data_merge.loc[:, 'date'].unique())
    for region in data_merge.loc[:, 'region'].unique():
        region_df = data_merge.loc[data_merge['region'] == region]
        region_train = pd.DataFrame()

        for n in range(number_of_days_in_one_row, number_of_days + 1):
            region_df_stack: pd.Series = region_df.iloc[
                                         (n - number_of_days_in_one_row):n,
                                         first_n_attribute_dsc_region:].stack().reset_index(drop=True)

            region_train = region_train.append(region_df_stack, ignore_index=True)
            n += 1

        first_index = number_of_days_in_one_row - 1
        region_df_dsc = region_df.iloc[first_index:, :first_n_attribute_dsc_region].reset_index(drop=True)
        region_train = region_train.reset_index(drop=True)

        region_train = pd.concat(
            [region_df_dsc, region_train], axis=1, ignore_index=True)

        train_all = train_all.append(region_train, ignore_index=True)

    return train_all


def make_target(data_merge: pd.DataFrame, number_of_days_in_one_rows, number_day_ahead_to_predition):
    first_index = number_of_days_in_one_rows + number_day_ahead_to_predition - 1

    target_f = pd.DataFrame()
    for region in data_merge.loc[:, 'region'].unique():
        one_region_target: pd.DataFrame = data_merge[data_merge['region'] == region].iloc[first_index:, [0, 1, -1]]
        one_region_target = one_region_target.set_index([one_region_target.columns[0], one_region_target.columns[1]])
        target_f = target_f.append(one_region_target)

    target_f = target_f.astype(float)
    return target_f


def make_date_to_prediction(train_all: pd.DataFrame):
    date_to_pr = pd.DataFrame()

    for region in train_all.loc[:, 0].unique():
        region_df = train_all.loc[train_all[0] == region]
        date_to_pr = date_to_pr.append(region_df.iloc[-1, :])

    date_to_pr: pd.DataFrame = date_to_pr.set_index([date_to_pr.columns[0], date_to_pr.columns[1]])
    date_to_pr = date_to_pr.astype(float)

    return date_to_pr


def get_train_only_with_preparation_n_days_ahead(train_all: pd.DataFrame,
                                                 number_day_ahead_to_prediction):
    train_f = pd.DataFrame()
    for region in train_all.iloc[:, 0].unique():
        one_region = train_all.loc[train_all[0] == region].iloc[:-number_day_ahead_to_prediction, :]
        train_f = train_f.append(one_region)
    train_f: pd.DataFrame = train_f.set_index([train_all.columns[0], train_all.columns[1]])
    train_f = train_f.astype(float)
    return train_f


def get_train_target(data_merge: pd.DataFrame, train_all: pd.DataFrame,
                     number_of_days_in_one_rows, number_day_ahead_to_predition):
    train = get_train_only_with_preparation_n_days_ahead(train_all, number_day_ahead_to_predition)

    target = make_target(data_merge, number_of_days_in_one_rows, number_day_ahead_to_predition)

    return train, target


def one_hot_encode(df, col_name, first_n_attribute_dsc_region):
    col_dsc = first_n_attribute_dsc_region

    if df[col_name].dtype == np.dtype('object'):
        dummies = pd.get_dummies(df[col_name], prefix=col_name)
        df = pd.concat([df.iloc[:, :col_dsc], dummies, df.iloc[:, col_dsc:]], axis=1)

    first_n_attribute_dsc_region = col_dsc + len(df[col_name].unique())
    return df, first_n_attribute_dsc_region
