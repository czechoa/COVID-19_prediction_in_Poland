import pandas as pd
from RNN.model.simple_regresion import make_all, clear_model, make_future_submission
from prepare_data.merge.merge_all_data import get_all_merge_data_from_to
from datetime import datetime, timedelta

from prepare_data.merge.data_epidemic_situation_in_regions.prepare_data_epidemic_situation_in_regions import \
    get_test_respiration
from prepare_data.test_train.make_train_test_from_merge_data import \
    reshape_data_merge_to_get_train_period_of_time_history, make_date_to_prediction, get_train_target


def next_day(date: str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")


def make_prediction_n_days_ahead(data_all_days: pd.DataFrame, period_of_time=21, last_day_train='2021-03-20',
                                 day_ahead=31):
    data_to_last_day = get_all_merge_data_from_to(last_day=last_day_train, data=data_all_days)

    train_all = reshape_data_merge_to_get_train_period_of_time_history(data_to_last_day, period_of_time)

    test_to_predict = make_date_to_prediction(train_all)

    data_all_days['date'] = data_all_days['date'].astype(str)
    last_day_train = str(data_to_last_day['date'].unique()[-1])
    result_all = pd.DataFrame(columns=['date', 'region', 'Engaged_respirator',
                                       'prediction'])
    day = next_day(last_day_train)
    for day_ahead_to_predict in range(1, day_ahead + 1):
        train, target = get_train_target(data_to_last_day, train_all, period_of_time, day_ahead_to_predict)

        make_all(train, target)
        submission = make_future_submission(test_to_predict, day)
        clear_model()

        test_ahead: pd.DataFrame = get_test_respiration(data_all_days, day)

        if test_ahead.empty:
            result = submission
        else:
            result = submission.merge(test_ahead, on=['region','date'])

        result_all = result_all.append(result, ignore_index=True)
        day = next_day(day)
    result_all = result_all.sort_values(by=['region', 'date'])

    return result_all
