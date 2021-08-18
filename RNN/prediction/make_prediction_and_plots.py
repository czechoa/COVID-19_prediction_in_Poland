from RNN.prediction.make_prediction_n_days_ahead import make_prediction_n_days_ahead
from RNN.prediction.plot.plots import plot_prediction_to_poland_from_results, subplot_prediction_for_all_region, \
    subplot_relative_error_for_all_region, plot_averaged_relative_error_for_all_region, plot_relative_error_for_poland
from prepareData.merge.merge_all_data import get_all_merge_data_from_to, get_all_merge_data

import pandas as pd


def make_prediction_and_subplot_for_all_regions(last_day_train='2021-03-20', day_ahead=30, period_of_time_learning=14,
                                                subplot_for_regions=True, relative_error=True):
    data_all_days_org = get_all_merge_data()

    data_all_days = data_all_days_org[data_all_days_org["region"] != 'POLSKA']

    results = make_prediction_n_days_ahead(data_all_days, day_ahead=day_ahead,
                                           period_of_time=period_of_time_learning, last_day_train=last_day_train)
    if subplot_for_regions:
        data_merge_from_to = get_all_merge_data_from_to(data= data_all_days, last_day= last_day_train)
        subplot_prediction_for_all_region([results], ['prediction'], data_merge_from_to)

    results['region'] = results['region'].replace('ŚŚ_average', 'POLSKA')
    results.loc[results['region'] == 'POLSKA', 'prediction'] = results.loc[
                                                                   results['region'] == 'POLSKA', 'prediction'] * 16
    plot_prediction_to_poland_from_results([results], ['prediction'], data_all_days_org)

    if relative_error:
        make_plots_relative_error_for_regions(results)

    return results


def make_plots_relative_error_for_regions(prediction=None):
    if prediction is not None:
        prediction = pd.read_csv('results/csv/prediction_for_region.csv')

    prediction['relative_error_%'] = 100 * abs(
        prediction['Engaged_respirator'] - prediction['prediction']) / prediction[
                                         'Engaged_respirator']
    prediction_Poland = prediction[prediction['region'] == 'ŚŚ_average']
    subplot_relative_error_for_all_region(prediction.copy())
    prediction = prediction[prediction['region'].isin(prediction['region'].unique()[:-3])]

    plot_averaged_relative_error_for_all_region(prediction)
    plot_relative_error_for_poland(prediction_Poland)
