from code.plot import plot_prediction_to_Poland_from_results, subplot_prediction_for_all_region
from code.model.simple_regresion import make_all, make_submission, clear_model
from code.prepareData.merge.merge_data_mobility_epidemic_situation import get_merge_data_from_to
# from prepareData.prepare_data_epidemic_situation_in_regions import get_test_respiration
from datetime import datetime, timedelta


# from prepareData.data_augmentation import data_augmentation

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


def make_list_results_by_averaged_from_1_3_7_days_back(data_merge_org_f: pd.DataFrame, period_of_time_f,
                                                       last_day_train_f):
    list_results = list()
    labels = list()
    for i in [1, 3, 7]:
        data_merge = averaged_merge_data_from_n_days(data_merge_org_f.copy(), i)
        result_all, result_all_err = make_prediction_one_month_ahead_for_train_all(data_merge, period_of_time_f,
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


def make_plot_for_Poland(list_results, labels, data_merge_from_to=get_merge_data_from_to(last_day='2021-05-01'),
                         title='Poland engaged respiration', save=False):
    plot_prediction_to_Poland_from_results(list_results, labels, data_merge_from_to, path='results/' + title,
                                           title=title)
    if save:
        save_list_results(list_results, path='results/' + title)


def make_prediction_with_data_augmentation_average_10():
    last_day_train = '2021-03-20'
    period_of_time = 21
    data_merge_org = get_merge_data_from_to(last_day=last_day_train)

    data_merge_org = data_merge_org[data_merge_org["region"] != 'POLSKA']
    # data_merge_org = data_augmentation(data_merge_org)
    sum_result_all = pd.DataFrame()
    for i in range(0, 10):
        results, results_error = make_prediction_one_month_ahead_for_train_all(data_merge_org, day_ahead=31)
        if i == 0:
            sum_result_all = results
        else:
            sum_result_all = sum_result_all + results

    sum_result_all.iloc[:, -1] = sum_result_all.iloc[:, -1].div(10)
    sum_result_all['date'] = results['date']
    sum_result_all['region'] = results['region']
    sum_result_all.to_csv('results/averaged_from_ten_prediction.csv', index=False)
    return sum_result_all
    # make_plot_for_Poland([sum_result_all], ['prediction'], title='Poland prediction average of 10 measurements',
    #                      data_merge_from_to=data_merge_to_2021_05, save=True)


def make_data_merge_from_to_from_last_day_train(last_day_train, days_ahead_to_prediction, delta):
    date = datetime.strptime(last_day_train, "%Y-%m-%d")

    modified_date = date - timedelta(days=delta)
    first_day = datetime.strftime(modified_date, "%Y-%m-%d")

    modified_date = date + timedelta(days=delta + days_ahead_to_prediction)
    last_day = datetime.strftime(modified_date, "%Y-%m-%d")

    data_merge_from_to = get_merge_data_from_to(first_day, last_day)
    return data_merge_from_to


def make_prediction_and_subplot_for_all_regions(subplot=True):
    last_day_train = '2021-03-20'
    period_of_time = 14
    data_merge_org = get_merge_data_from_to(last_day=last_day_train)

    data_merge_org = data_merge_org[data_merge_org["region"] != 'POLSKA']
    # data_merge_org = data_augmentation(data_merge_org)
    # list_results, labels = make_list_results_by_averaged_from_1_3_7_days_back(data_merge_org, period_of_time, last_day_train)
    results, results_error = make_prediction_one_month_ahead_for_train_all(data_merge_org, day_ahead=30,
                                                                           period_of_time=period_of_time)

    if subplot:
        results.to_csv('results/prediction_for_region_with_data_augmentation.csv', index=False)

        data_merge_from_to = get_merge_data_from_to(last_day='2021-05-05')
        # subplot_prediction_for_all_region(list_results, labels, data_merge_from_to)
        subplot_prediction_for_all_region([results], ['prediction'], data_merge_from_to)
    results['region'] = results['region'].replace('ŚŚ_average', 'POLSKA')
    # results[results['region'] == 'POLSKA'].iloc[:, [-2, -1]] = results[results['region'] == 'POLSKA'].iloc[:,
    #                                                            [-2, -1]] * 16
    results.iloc[:, -1] = results.iloc[:, -1] * 16
    make_plot_for_Poland([results], ['prediction'], )
    plot_prediction_to_Poland_from_results([results], ['prediction'], get_merge_data_from_to(last_day='2021-05-05'))

    # save_list_results(list_results)
    return results


# %%
# data_merge_from_to = make_data_merge_from_to_from_last_day_train('2021-03-20', 31, 21)
result_14 = make_prediction_and_subplot_for_all_regions(subplot=True)

# %%
data_merge_all = get_merge_data_from_to(last_day='2021-05-05')
date = '2021-03-21'
data_merge_all = data_merge_all.iloc[:, [0, 1, -1]]
print(date)
data_merge_all['date'] = data_merge_all['date'].astype(str)

finale_day = data_merge_all[data_merge_all['date'] == date]

# w = data_merge_all['date'].unique()
# %%
# make_prediction_and_subplot_for_all_regions()
# results = make_prediction_and_subplot_for_all_regions()

# results = make_prediction_with_data_augmentation_average_10
# subplot_prediction_for_all_region(list_results, labels, data_merge_from_to)
# subplot_prediction_for_all_region(list_results, labels, data_merge_from_to)
period_of_time = 21
last_day_train = '2021-03-20'
data_merge = get_merge_data_from_to(last_day=last_day_train)
day_ahead_to_predict = 1
train_all = reshape_data_merge_to_get_train_period_of_time_history(data_merge, period_of_time)

test_to_predict = make_date_to_prediction(train_all)
data_merge_all = get_merge_data_from_to()
data_merge_all['date'] = data_merge_all['date'].astype(str)
# train_all = standardScaler(train_all,test_to_predict)

result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
                                   'prediction'])
result_all_err = pd.DataFrame()
day = next_day(last_day_train)
train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)
# train,test_to_predict = standardScaler(train,test_to_predict)

make_all(train, target)
submission = make_submission(test_to_predict, day_ahead_to_predict)
clear_model()
submission = submission.reset_index()
test_ahead: pd.DataFrame = get_test_respiration(data_merge_all, day)
# %%
submission.rename(columns={submission.columns[0]: test_ahead.columns[0], submission.columns[1]: test_ahead.columns[1],
                           submission.columns[2]: 'prediction'},
                  inplace=True)
submission = submission.drop(columns='date')
# %%
result = test_ahead.merge(submission, on=['region'])
# %%
import pandas as pd

results = pd.read_csv('../../results/prediction_for_region_with_data_augmentation.csv')
# %%
results['region'] = results['region'].replace('ŚŚ_average', 'POLSKA')
# results[results['region'] == 'POLSKA'].iloc[:, [-2, -1]] = results[results['region'] == 'POLSKA'].iloc[:,
#                                                            [-2, -1]] * 16
w = results[results['region'] == 'POLSKA'].iloc[:, [-1]] * 16
results.iloc[:, -1] = results.iloc[:, -1] * 16
make_plot_for_Poland([results], ['prediction'], )
plot_prediction_to_Poland_from_results([results], ['prediction'], get_merge_data_from_to(last_day='2021-05-05'))
