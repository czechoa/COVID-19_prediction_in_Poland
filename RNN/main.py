from RNN.prediction.make_prediction_and_plots import make_prediction_and_subplot_for_all_regions, \
    make_plots_relative_error_for_regions
# %% las day train 20 march
result = make_prediction_and_subplot_for_all_regions(day_ahead=30,period_of_time=7)
make_plots_relative_error_for_regions(prediction = result)

# %% now
result = make_prediction_and_subplot_for_all_regions(last_day_train= None,day_ahead=30,period_of_time=7)
make_plots_relative_error_for_regions(prediction = result)






