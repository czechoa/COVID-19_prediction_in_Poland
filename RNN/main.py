from RNN.prediction.make_prediction_and_plots import make_prediction_and_subplot_for_all_regions

result = make_prediction_and_subplot_for_all_regions(last_day_train= '2021-03-30', day_ahead=2, period_of_time_learning=7,relative_error= False)

# %% now
result = make_prediction_and_subplot_for_all_regions(last_day_train= None, day_ahead=30, period_of_time_learning=7, subplot_for_regions= False, relative_error=False)






