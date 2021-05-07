from prepare_data_mobility import get_prepared_data_mobility
from prepare_data_epidemic_situation_in_regions import prepare_data_epidemic_situation_in_regions
import pandas as pd
import numpy as np
# %%
def get_data_mobility_data_epidemic_situation_in_regions():
    data_mobility = get_prepared_data_mobility()
    data_epidemic_situation_in_regions = prepare_data_epidemic_situation_in_regions()
    return data_mobility, data_epidemic_situation_in_regions


# %%
def change_to_polish_name_regions_in_mobility_data(data_mobility: pd.DataFrame,
                                                   data_epidemic_situation_in_regions: pd.DataFrame):
    data_epidemic_situation_in_regions.sort_values(by='region')
    region = data_epidemic_situation_in_regions.loc[:, 'region'].unique()
    region = list(region)
    data_mobility.sort_values(by='sub_region_1')
    region_mobility = data_mobility.loc[:, 'sub_region_1'].unique()
    region_mobility = list(region_mobility)
    region_df = pd.DataFrame()
    region_df['mobility'] = region_mobility
    region_df['region_epidemic_situation'] = region
    myorder = [14, 1, 5, 0, 2, 3, 6, 7, 8, 9, 10, 11, 13, 15, 4, 12]
    region = [region[i] for i in myorder]
    region_df['region_epidemic_situation'] = region
    number_date = data_mobility.shape[0] / len(region_mobility)
    region_repeat = np.repeat(region, number_date)
    data_mobility.insert(1, 'region', region_repeat)
    return data_mobility, data_epidemic_situation_in_regions


# %%
def merge_data_mobility_covid_19_situation(data_mobility: pd.DataFrame,
                                           data_epidemic_situation_in_regions: pd.DataFrame):
    data_merge = data_mobility.merge(data_epidemic_situation_in_regions, how='inner', on=['region', 'date'])
    data_merge: pd.DataFrame = data_merge.drop(columns=[data_merge.columns[0]])
    return data_merge


# %%
def get_merge_data():

    data_mobility, data_epidemic_situation_in_regions = get_data_mobility_data_epidemic_situation_in_regions()
    data_mobility, data_epidemic_situation_in_regions = change_to_polish_name_regions_in_mobility_data(data_mobility,
                                                                                                       data_epidemic_situation_in_regions)
    data_merge = merge_data_mobility_covid_19_situation(data_mobility, data_epidemic_situation_in_regions)
    return data_merge
# %%
# merge  = get_merge_data()
# merge.to_csv('{}.csv'.format('results/data_merge'), index=False)
#

