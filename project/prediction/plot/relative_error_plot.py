import pandas as pd

from project.prediction.plot.plots import subplot_relative_error_for_all_region, plot_averaged_relative_error_for_all_region, \
    plot_relative_error_for_Polska

def make_plots_relative_error_for_regions(prediction = None):
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


