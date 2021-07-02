from project.model.make_prediction import make_prediction_and_subplot_for_all_regions
from project.plot.relative_error_plot import make_plots_relative_error_for_regions_with_augmentation

result = make_prediction_and_subplot_for_all_regions()
make_plots_relative_error_for_regions_with_augmentation(prediction = result)