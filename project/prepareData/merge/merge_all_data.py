from project.prepareData.merge.merge_data_mobility_epidemic_situation import get_merge_data_from_to
from project.prepareData.augmentation.data_augmentation import data_augmentation
from project.prepareData.prepare_data_area_population import preparing_data_area_population_regions
import pandas as pd


def get_all_merge_data_from_to(first_day: str = None, last_day='2021-05-05'):
    merge_data = get_merge_data_from_to(first_day, last_day)
    merge_from_to = merge_area_population(merge_data)
    merge_from_to = data_augmentation(merge_from_to)

    return merge_from_to


def merge_area_population(merge_data: pd.DataFrame, attribute_dsc=3):
    region_area_population: pd.DataFrame = preparing_data_area_population_regions()

    merge_dsc: pd.DataFrame = merge_data.iloc[:, :attribute_dsc].merge(region_area_population, on='region')
    merge_all = merge_dsc.merge(merge_data, on=list(merge_data.columns.values[0:attribute_dsc]))

    return merge_all
# %%
# from project.prepareData.test_train.make_train_test_from_merge_data import one_hot_encode
# a = get_all_merge_data_from_to()
# b, number_desc =  one_hot_encode(a,'region',5)
# b.to_csv('results/csv/data_all_with_one_hot_encode.csv', index=False)

