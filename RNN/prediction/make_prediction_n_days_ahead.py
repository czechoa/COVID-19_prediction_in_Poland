import pandas as pd
from RNN.model.simple_regresion import make_all, make_submission, clear_model, make_future_submission
from prepareData.merge.merge_all_data import get_all_merge_data_from_to
from datetime import datetime, timedelta

from prepareData.prepare_data_epidemic_situation_in_regions import get_test_respiration
from prepareData.test_train.make_train_test_from_merge_data import \
    reshape_data_merge_to_get_train_period_of_time_history, make_date_to_prediction, get_train_target


def next_day(date: str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")


def make_prediction_n_days_ahead(data_all_days:pd.DataFrame, period_of_time=21, last_day_train='2021-03-20',
                                 day_ahead=31):

    data_to_last_day = get_all_merge_data_from_to(last_day=last_day_train,data=data_all_days)

    train_all = reshape_data_merge_to_get_train_period_of_time_history(data_to_last_day, period_of_time)

    test_to_predict = make_date_to_prediction(train_all)

    # data_all_days = get_all_merge_data_from_to(last_day=None)
    # data_merge_all:pd.DataFrame = data_merge.copy()
    data_all_days['date'] = data_all_days['date'].astype(str)
    # train_all = standardScaler(train_all,test_to_predict)
    last_day_train = str(data_to_last_day['date'].unique()[-1])
    result_all = pd.DataFrame(columns=['date', 'region', 'Engaged_respirator',
                                       'prediction'])
    print(last_day_train)
    # result_all_err = pd.DataFrame()
    day = next_day(last_day_train)
    for day_ahead_to_predict in range(1, day_ahead + 1):
        train, target = get_train_target(data_to_last_day, train_all, period_of_time, day_ahead_to_predict)
        # train,test_to_predict = standardScaler(train,test_to_predict)

        make_all(train, target)
        submission = make_future_submission(test_to_predict, day)
        clear_model()

        # submission = submission.reset_index()
        test_ahead: pd.DataFrame = get_test_respiration(data_all_days, day)

        # submission.rename(
        #     columns={submission.columns[0]: test_ahead.columns[0], submission.columns[1]: test_ahead.columns[1],
        #              submission.columns[2]: 'prediction'},
        #     inplace=True)
        # submission = submission.drop(columns='date')

        if test_ahead.empty:
            result = submission
        else:
            submission = submission.drop(columns='date')
            result = submission.merge(test_ahead, on=['region'])

        # result_err = result.iloc[:, :2]
        # result_err['subtract'] = result.iloc[:, -2].astype(float) - result.iloc[:, -1].astype(float)
        # result_err['relative error in %'] = abs(result_err.loc[:, 'subtract'] / result.iloc[:, -1].astype(float)) * 100
        result_all = result_all.append(result, ignore_index=True)
        # result_all_err = result_all_err.append(result_err, ignore_index=True)
        day = next_day(day)
    # print(day_ahead_to_predict)
    result_all = result_all.sort_values(by=['region', 'date'])
    # result_all_err = result_all_err.sort_values(by=['region', 'date'])

    return result_all
