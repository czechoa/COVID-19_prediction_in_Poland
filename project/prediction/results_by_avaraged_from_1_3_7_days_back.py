from project.prediction.make_prediction_one_month_ahead import make_prediction_one_month_ahead_for_train_all
from project.prediction.plot.plots import plot_prediction_to_Poland_from_results
from project.prepareData.test_train.make_train_test_from_merge_data import averaged_merge_data_from_n_days


def save_list_results(list_results, path='results/csv/all_prediction_for_region.csv'):
    all_prediction = pd.DataFrame()
    for result_all_it, i in zip(list_results.copy(), list([1, 3, 7])):
        result_all_it.insert(2, 'avarage from n days back', str(i))
        all_prediction = all_prediction.append(result_all_it, ignore_index=True)
        result_all_it.drop(columns='avarage from n days back', inplace=True)
    all_prediction.to_csv(path, index=False)


def from_all_prection_to_list_results_and_labels(all_prediction: pd.DataFrame):
    list_results = list()
    labels = list()

    for avaraged_from_n in all_prediction['avarage from n days back'].unique():
        result = all_prediction[all_prediction['avarage from n days back'] == avaraged_from_n]
        result = result.drop(columns='avarage from n days back')
        list_results.append(result)
        label = 'prediction from averaged ' + str(avaraged_from_n) + ' days back'
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

def make_plot_for_Poland(list_results, labels, data_merge_from_to=get_merge_data_from_to(last_day='2021-05-01'),
                         title='Poland engaged respiration', save=False):
    plot_prediction_to_Poland_from_results(list_results, labels, data_merge_from_to, path='results/' + title,
                                           title=title)
    if save:
        save_list_results(list_results, path='results/' + title)
