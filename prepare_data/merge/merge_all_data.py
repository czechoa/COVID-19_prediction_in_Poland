from prepare_data.merge.data_area_and_population.prepare_data_area_population import \
    preparing_data_area_population_regions
from prepare_data.merge.merge_data_mobility_epidemic_situation import get_merge_data_from_to, get_merge_data
from prepare_data.augmentation.data_augmentation import data_augmentation
import pandas as pd
from prepare_data.test_train.make_train_test_from_merge_data import one_hot_encode


def get_all_merge_data_from_to(first_day: str = None, last_day='2021-05-05', number_of_gaussian_noise_regions=1,
                               data=None):
    if data is None:
        data = get_all_merge_data(number_of_gaussian_noise_regions)
    data_all_from_to = get_merge_data_from_to(first_day, last_day, data)

    return data_all_from_to

def get_all_merge_data(number_of_gaussian_noise_regions=1):
    merge_data_all_days = get_merge_data()
    merge_data_all_days = merge_area_population(merge_data_all_days)
    merge_data_all_days = data_augmentation(merge_data_all_days, number_of_gaussian_noise_regions)
    merge_data_all_days = merge_data_all_days.sort_values(by=['region', 'date'])
    return merge_data_all_days


def merge_area_population(merge_data: pd.DataFrame, attribute_dsc=3):
    region_area_population: pd.DataFrame = preparing_data_area_population_regions()

    merge_dsc: pd.DataFrame = merge_data.iloc[:, :attribute_dsc].merge(region_area_population, on='region')
    merge_all = merge_dsc.merge(merge_data, on=list(merge_data.columns.values[0:attribute_dsc]))

    return merge_all


def save_all_merge_data_with_one_hot_encode():
    merge_data = get_all_merge_data_from_to(number_of_gaussian_noise_regions=10)
    merge_data_one_hot_encode, number_desc = one_hot_encode(merge_data, 'region', 5)
    merge_data_one_hot_encode.to_csv('data/data_lstm/data_all_with_one_hot_encode.csv', index=False)


# a = get_all_merge_data()
# b = get_all_merge_data_from_to(data=a)
