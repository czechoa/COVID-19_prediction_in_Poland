from project.model.simple_regresion import make_all, make_submission, clear_model
from project.prepareData.merge.merge_data_mobility_epidemic_situation import get_merge_data_from_to
from datetime import datetime, timedelta


# from prepareData.data_augmentation import data_augmentation
from project.prepareData.test_train.make_train_test_from_merge_data import \
    reshape_data_merge_to_get_train_period_of_time_history, make_date_to_prediction, get_train_target


def next_day(date: str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")


def get_test_respiration(data_merge: pd.DataFrame, date):
    data_merge = data_merge.iloc[:, [0, 1, -1]]
    finale_day = data_merge[data_merge['date'] == date]
    return finale_day


def make_prediction_one_month_ahead_for_train_all(data_merge, period_of_time=21, last_day_train='2021-03-20',
                                                  day_ahead=31):
    train_all = reshape_data_merge_to_get_train_period_of_time_history(data_merge, period_of_time)

    test_to_predict = make_date_to_prediction(train_all)
    data_merge_all = get_merge_data_from_to(last_day='2021-05-05')
    data_merge_all['date'] = data_merge_all['date'].astype(str)
    # train_all = standardScaler(train_all,test_to_predict)

    result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
                                       'prediction'])
    result_all_err = pd.DataFrame()
    day = next_day(last_day_train)
    for day_ahead_to_predict in range(1, day_ahead + 1):
        train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)
        # train,test_to_predict = standardScaler(train,test_to_predict)

        make_all(train, target)
        submission = make_submission(test_to_predict, day_ahead_to_predict)
        clear_model()

        submission = submission.reset_index()
        test_ahead: pd.DataFrame = get_test_respiration(data_merge_all, day)

        submission.rename(
            columns={submission.columns[0]: test_ahead.columns[0], submission.columns[1]: test_ahead.columns[1],
                     submission.columns[2]: 'prediction'},
            inplace=True)
        submission = submission.drop(columns='date')

        result = test_ahead.merge(submission, on=['region'])

        result_err = result.iloc[:, :2]
        result_err['subtract'] = result.iloc[:, -2].astype(float) - result.iloc[:, -1].astype(float)
        result_err['relative error in %'] = abs(result_err.loc[:, 'subtract'] / result.iloc[:, -1].astype(float)) * 100
        result_all = result_all.append(result, ignore_index=True)
        result_all_err = result_all_err.append(result_err, ignore_index=True)
        day = next_day(day)
    print(day_ahead_to_predict)
    result_all = result_all.sort_values(by=['region', 'date'])
    result_all_err = result_all_err.sort_values(by=['region', 'date'])

    return result_all, result_all_err



# %%
# data_merge_from_to = make_data_merge_from_to_from_last_day_train('2021-03-20', 31, 21)
# result_14 = make_prediction_and_subplot_for_all_regions(subplot=True)
#
# # %%
# data_merge_all = get_merge_data_from_to(last_day='2021-05-05')
# date = '2021-03-21'
# data_merge_all = data_merge_all.iloc[:, [0, 1, -1]]
# print(date)
# data_merge_all['date'] = data_merge_all['date'].astype(str)
#
# finale_day = data_merge_all[data_merge_all['date'] == date]

# w = data_merge_all['date'].unique()
# %%
# make_prediction_and_subplot_for_all_regions()
# results = make_prediction_and_subplot_for_all_regions()

# results = make_prediction_with_data_augmentation_average_10
# subplot_prediction_for_all_region(list_results, labels, data_merge_from_to)
# subplot_prediction_for_all_region(list_results, labels, data_merge_from_to)
# period_of_time = 21
# last_day_train = '2021-03-20'
# data_merge = get_merge_data_from_to(last_day=last_day_train)
# day_ahead_to_predict = 1
# train_all = reshape_data_merge_to_get_train_period_of_time_history(data_merge, period_of_time)
#
# test_to_predict = make_date_to_prediction(train_all)
# data_merge_all = get_merge_data_from_to()
# data_merge_all['date'] = data_merge_all['date'].astype(str)
# # train_all = standardScaler(train_all,test_to_predict)
#
# result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
#                                    'prediction'])
# result_all_err = pd.DataFrame()
# day = next_day(last_day_train)
# train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)
# # train,test_to_predict = standardScaler(train,test_to_predict)
#
# make_all(train, target)
# submission = make_submission(test_to_predict, day_ahead_to_predict)
# clear_model()
# submission = submission.reset_index()
# test_ahead: pd.DataFrame = get_test_respiration(data_merge_all, day)
# # %%
# submission.rename(columns={submission.columns[0]: test_ahead.columns[0], submission.columns[1]: test_ahead.columns[1],
#                            submission.columns[2]: 'prediction'},
#                   inplace=True)
# submission = submission.drop(columns='date')
# # %%
# result = test_ahead.merge(submission, on=['region'])
# # %%
# import pandas as pd
#
# results = pd.read_csv('../../results/prediction_for_region_with_data_augmentation.csv')
# # %%
# results['region'] = results['region'].replace('ŚŚ_average', 'POLSKA')
# # results[results['region'] == 'POLSKA'].iloc[:, [-2, -1]] = results[results['region'] == 'POLSKA'].iloc[:,
# #                                                            [-2, -1]] * 16
# w = results[results['region'] == 'POLSKA'].iloc[:, [-1]] * 16
# results.iloc[:, -1] = results.iloc[:, -1] * 16
# make_plot_for_Poland([results], ['prediction'], )
# plot_prediction_to_Poland_from_results([results], ['prediction'], get_merge_data_from_to(last_day='2021-05-05'))
