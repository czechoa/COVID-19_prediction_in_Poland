from prepareData.merge_data_mobility_epidemic_situation import get_merge_data_from_to
from prepareData.data_augmentation import data_augmentation
from prepareData.prepare_data_area_population import preparing_data_area_population_regions
import pandas as pd

def get_all_merge_data_from_to(first_day: str = None, last_day='2021-04-04'):

    merge_data = get_merge_data_from_to(first_day, last_day)
    merge_from_to = merge_area_population(merge_data)
    merge_from_to = data_augmentation(merge_from_to)

    return merge_from_to

def merge_area_population(merge_data: pd.DataFrame):
    region_area_population: pd.DataFrame = preparing_data_area_population_regions()

    merge_dsc: pd.DataFrame = merge_data.iloc[:, :3].merge(region_area_population, on='region')
    merge_all = merge_dsc.merge(merge_data, on=list(merge_data.columns.values[0:3]))

    return merge_all
# %%
# a = get_all_merge_data_from_to(last_day='2021-05-01')
