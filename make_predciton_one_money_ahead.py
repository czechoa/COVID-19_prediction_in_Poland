from make_train_test_from_merge_data import *
from plots import plot_prediction_to_Poland_from_results, subplot_prediction_for_all_region
from simple_regresion import make_all, make_submission, clear_model
from merge_data_mobility_epidemic_situation import get_merge_data_from_to
from prepare_data_epidemic_situation_in_regions import get_test_respiration
from datetime import datetime, timedelta


def next_day(date: str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")


def make_prediction_one_mounth_ahead_for_train_all(data_merge, period_of_time=21, last_day_train='2021-03-20'):
    train_all = reshape_data_merge_to_get_train_period_of_time_history_1(data_merge, period_of_time)

    test_to_predict = make_date_to_prediction(train_all)

    result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
                                       'prediction'])
    result_all_err = pd.DataFrame()
    day = next_day(last_day_train)
    for day_ahead_to_predict in range(1, 31):
        train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)

        make_all(train, target)
        submission = make_submission(test_to_predict, day_ahead_to_predict)
        clear_model()

        submission = submission.reset_index()
        test_ahead: pd.DataFrame = get_test_respiration(date=day)

        submission.rename(columns={submission.columns[0]: test_ahead.columns[1], submission.columns[2]: 'prediction'},
                          inplace=True)
        result = pd.merge(test_ahead, submission.drop(columns=submission.columns[1]), on=test_ahead.columns[1])
        result_err = result.iloc[:, :2]
        result_err['subtract'] = result.iloc[:, -2].astype(float) - result.iloc[:, -1].astype(float)
        result_err['relative error in %'] = abs(result_err.loc[:, 'subtract'] / result.iloc[:, -1].astype(float)) * 100
        result_all = result_all.append(result, ignore_index=True)
        result_all_err = result_all_err.append(result_err, ignore_index=True)
        day = next_day(day)

    result_all = result_all.sort_values(by=['region', 'date'])
    result_all_err = result_all_err.sort_values(by=['region', 'date'])

    return result_all, result_all_err


def save_list_results(list_results, path='results/all_prediction_for_region.csv'):
    all_prediction = pd.DataFrame()
    for result_all_it, i in zip(list_results.copy(), list([1, 3, 7])):
        result_all_it.insert(2, 'avarage from n days back', str(i))
        all_prediction = all_prediction.append(result_all_it, ignore_index=True)
        result_all_it.drop(columns='avarage from n days back', inplace=True)
    all_prediction.to_csv(path, index=False)


def from_all_prection_to_list_results_and_labels(all_prediction: pd.DataFrame):
    list_results = list()
    labels = list()

    for avarage_from_n in all_prediction['avarage from n days back'].unique():
        result = all_prediction[all_prediction['avarage from n days back'] == avarage_from_n]
        result = result.drop(columns='avarage from n days back')
        list_results.append(result)
        label = 'prediction from averaged ' + str(avarage_from_n) + ' days back'
        labels.append(label)

    return list_results, labels


def add_region_Poland_as_mean_groupby_date(data_merge_f: pd.DataFrame):
    data_merge_f = data_merge_f[data_merge_f["region"] != 'POLSKA']
    poland_grub: pd.DataFrame = data_merge_f.groupby(by='date').mean().reset_index()
    poland_grub.insert(0, 'region', 'POLSKA')
    data_merge_f = data_merge_f.append(poland_grub, ignore_index=True)
    data_merge_f = data_merge_f.sort_values(by=['region', 'date'])
    return data_merge_f


def make_list_results_by_averaged_from_1_3_7_days_back(data_merge_org_f: pd.DataFrame, period_of_time_f,
                                                       last_day_train_f):
    list_results = list()
    labels = list()
    for i in [1, 3, 7]:
        data_merge = avarage_merge_data_from_n_days(data_merge_org_f.copy(), i)
        result_all, result_all_err = make_prediction_one_mounth_ahead_for_train_all(data_merge, period_of_time_f,
                                                                                    last_day_train_f)
        label = 'prediction from averaged ' + str(i) + ' days back'
        list_results.append(result_all)
        labels.append(label)
    return list_results, labels


def make_plot_Poland_as_groupBy(list_results_f: pd.DataFrame, labels, data_merge_from_to, save=False):
    # TODO mul only POLAND by *16
    for result_all_it in list_results_f:
        result_all_it['prediction'] = result_all_it['prediction'] * 16

    title = 'Poland engaged respiration, learned by mean Poland'

    make_plot_for_Poland(list_results_f, labels, data_merge_from_to, title, save)


def make_plot_for_Poland(list_results, labels, data_merge_from_to, title='Poland engaged respiration', save=False):
    plot_prediction_to_Poland_from_results(list_results, labels, data_merge_from_to, path='results/' + title,
                                           title=title)
    if save:
        save_list_results(list_results, path='results/' + title)


def make_prediction_and_subplot_for_all_regions():
    last_day_train = '2021-03-20'
    period_of_time = 21
    data_merge_org = get_merge_data_from_to(last_day=last_day_train)
    data_merge_org = data_merge_org[data_merge_org["region"] != 'POLSKA']
    list_results, labels = make_list_results_by_averaged_from_1_3_7_days_back(data_merge_org, period_of_time,
                                                                              last_day_train)
    data_merge_from_to = get_merge_data_from_to('2021-03-01', '2021-05-01')
    subplot_prediction_for_all_region(list_results, labels, data_merge_from_to)


def make_data_merge_from_to_from_last_day_train(last_day_train, days_ahead_to_prediction, delta):
    date = datetime.strptime(last_day_train, "%Y-%m-%d")

    modified_date = date - timedelta(days=delta)
    first_day = datetime.strftime(modified_date, "%Y-%m-%d")

    modified_date = date + timedelta(days=delta + days_ahead_to_prediction)
    last_day = datetime.strftime(modified_date, "%Y-%m-%d")

    data_merge_from_to = get_merge_data_from_to(first_day, last_day)
    return data_merge_from_to


# %%
# data_merge_from_to = make_data_merge_from_to_from_last_day_train('2021-03-20', 31, 21)
