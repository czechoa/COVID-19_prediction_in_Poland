import pandas as pd
from code.plots import subplot_relative_error_for_all_region, plot_averaged_relative_error_for_all_region, \
    plot_relative_error_for_Polska


def make_plots_relative_error_for_regions():
    all_prediction_list = pd.read_csv('../../results/all_prediction_for_region.csv')
    prediction = all_prediction_list[all_prediction_list['avarage from n days back'] == 1]
    prediction['relative_error_%'] = 100 * abs(
        prediction['Liczba zajętych respiratorów (stan ciężki)'] - prediction['prediction']) / prediction[
                                         'Liczba zajętych respiratorów (stan ciężki)']
    subplot_relative_error_for_all_region(prediction)
    plot_averaged_relative_error_for_all_region(prediction)


def make_plots_relative_error_for_regions_with_augumetation():
    prediction = pd.read_csv('../../results/prediction_for_region_with_data_augmentation.csv')
    prediction['relative_error_%'] = 100 * abs(
        prediction['Liczba zajętych respiratorów (stan ciężki)'] - prediction['prediction']) / prediction[
                                         'Liczba zajętych respiratorów (stan ciężki)']
    prediction_Poland = prediction[prediction['region'] == 'ŚŚ_average']
    subplot_relative_error_for_all_region(prediction.copy())
    prediction = prediction[prediction['region'].isin(prediction['region'].unique()[:-3])]
    plot_averaged_relative_error_for_all_region(prediction)
    plot_relative_error_for_Polska(prediction_Poland)


# %%
make_plots_relative_error_for_regions_with_augumetation()
