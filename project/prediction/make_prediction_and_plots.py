from datetime import datetime, timedelta

from project.prediction.make_prediction_n_days_ahead import make_prediction_n_days_ahead
from project.prediction.plot.plots import plot_prediction_to_Poland_from_results, subplot_prediction_for_all_region, \
    subplot_relative_error_for_all_region, plot_averaged_relative_error_for_all_region, plot_relative_error_for_Polska
from project.prepareData.merge.merge_data_mobility_epidemic_situation import get_merge_data_from_to

import pandas as pd


def make_data_merge_from_to_from_last_day_train(last_day_train, days_ahead_to_prediction, delta):
    date = datetime.strptime(last_day_train, "%Y-%m-%d")

    modified_date = date - timedelta(days=delta)
    first_day = datetime.strftime(modified_date, "%Y-%m-%d")

    modified_date = date + timedelta(days=delta + days_ahead_to_prediction)
    last_day = datetime.strftime(modified_date, "%Y-%m-%d")

    data_merge_from_to = get_merge_data_from_to(first_day, last_day)
    return data_merge_from_to


def make_prediction_and_subplot_for_all_regions(last_day_train='2021-03-20', day_ahead=30, period_of_time=14,
                                                subplot=True):
    data_merge_org = get_merge_data_from_to(last_day=last_day_train)

    data_merge_org = data_merge_org[data_merge_org["region"] != 'POLSKA']

    results, results_error = make_prediction_n_days_ahead(data_merge_org, day_ahead=day_ahead,
                                                          period_of_time=period_of_time)
    if subplot:
        results.to_csv('results/csv/prediction_for_region.csv', index=False)

        data_merge_from_to = get_merge_data_from_to(last_day='2021-05-05')
        subplot_prediction_for_all_region([results], ['prediction'], data_merge_from_to)
    results['region'] = results['region'].replace('ŚŚ_average', 'POLSKA')

    results.iloc[:, -1] = results.iloc[:, -1] * 16

    plot_prediction_to_Poland_from_results([results], ['prediction'], get_merge_data_from_to(last_day='2021-05-05'))

    return results


def make_plots_relative_error_for_regions(prediction=None):
    if prediction is not None:
        prediction = pd.read_csv('results/csv/prediction_for_region.csv')

    prediction['relative_error_%'] = 100 * abs(
        prediction['Liczba zajętych respiratorów (stan ciężki)'] - prediction['prediction']) / prediction[
                                         'Liczba zajętych respiratorów (stan ciężki)']
    prediction_Poland = prediction[prediction['region'] == 'ŚŚ_average']
    subplot_relative_error_for_all_region(prediction.copy())
    prediction = prediction[prediction['region'].isin(prediction['region'].unique()[:-3])]
    plot_averaged_relative_error_for_all_region(prediction)
    plot_relative_error_for_Polska(prediction_Poland)
