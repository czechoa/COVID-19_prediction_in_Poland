from RNN.prediction.make_prediction_and_plots import make_prediction_and_subplot_for_all_regions
# %% to 2021-03-25
result = make_prediction_and_subplot_for_all_regions(last_day_train= '2021-03-25', day_ahead=21, period_of_time_learning=7)

# %% current
result = make_prediction_and_subplot_for_all_regions(last_day_train= None, day_ahead=21, period_of_time_learning=7, subplot_for_regions= False, relative_error=False)






