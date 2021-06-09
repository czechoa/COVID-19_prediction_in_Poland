import pandas as pd
from plots import subplot_relative_error_for_all_region, plot_averaged_relative_error_for_all_region
def make_plots_relative_error_for_regions():
    all_prediction_list = pd.read_csv('results/all_prediction_for_region.csv')
    prediction = all_prediction_list[all_prediction_list['avarage from n days back'] == 1]
    prediction['relative_error_%'] = 100*abs(prediction['Liczba zajętych respiratorów (stan ciężki)'] - prediction['prediction'])/prediction['Liczba zajętych respiratorów (stan ciężki)']
    subplot_relative_error_for_all_region(prediction)
    plot_averaged_relative_error_for_all_region(prediction)
# %%
# make_plots_relative_error_for_regions()
