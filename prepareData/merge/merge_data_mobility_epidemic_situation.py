from datetime import datetime
import pandas as pd
import numpy as np

from prepareData.prepare_data_epidemic_situation_in_regions import prepare_data_epidemic_situation_in_regions
from prepareData.prepare_data_mobility import get_prepared_data_mobility


def get_data_mobility_and_data_epidemic_situation_in_regions():
    data_mobility = get_prepared_data_mobility()
    data_epidemic_situation_in_regions = prepare_data_epidemic_situation_in_regions()
    return data_mobility, data_epidemic_situation_in_regions


def change_to_polish_name_regions_in_mobility_data(data_mobility: pd.DataFrame,
                                                   data_epidemic_situation_in_regions: pd.DataFrame):
    data_epidemic_situation_in_regions = data_epidemic_situation_in_regions.sort_values(by='region')

    region = data_epidemic_situation_in_regions.loc[:, 'region'].unique()
    region = list(region)

    data_mobility = data_mobility.sort_values(by='sub_region_1')

    region_mobility = data_mobility['sub_region_1'].unique()
    region_mobility = list(region_mobility)
    region_df = pd.DataFrame()
    region_df['mobility'] = region_mobility
    region_df['region_epidemic_situation'] = region

    myorder = [12, 1, 5, 0, 2, 3, 4, 6, 7, 8, 9, 10, 15, 11, 13, 14, 16]

    region = [region[i] for i in myorder]
    region_df['region_epidemic_situation'] = region
    number_date = data_mobility.shape[0] / len(region_mobility)
    region_repeat = np.repeat(region, number_date)
    data_mobility.insert(1, 'region', region_repeat)

    return data_mobility, data_epidemic_situation_in_regions


def merge_data_mobility_covid_19_situation(data_mobility: pd.DataFrame,
                                           data_epidemic_situation_in_regions: pd.DataFrame):
    data_merge = data_mobility.merge(data_epidemic_situation_in_regions, how='inner', on=['region', 'date'])
    data_merge: pd.DataFrame = data_merge.drop(columns=[data_merge.columns[0]])
    return data_merge


def get_merge_data():
    data_mobility, data_epidemic_situation_in_regions = get_data_mobility_and_data_epidemic_situation_in_regions()
    data_mobility, data_epidemic_situation_in_regions = change_to_polish_name_regions_in_mobility_data(data_mobility,
                                                                                                       data_epidemic_situation_in_regions)
    data_merge = merge_data_mobility_covid_19_situation(data_mobility, data_epidemic_situation_in_regions)

    return data_merge


def get_merge_data_from_to(first_day: str = None, last_day='2021-04-04'):
    merge = get_merge_data()

    merge['date'] = pd.to_datetime(merge.loc[:, 'date'], format='%Y-%m-%d').dt.date

    if (first_day != None):
        merge = merge.loc[merge['date'] >= datetime.strptime(first_day, "%Y-%m-%d").date()]

    merge_from_to: pd.DataFrame = merge.loc[merge['date'] <= datetime.strptime(last_day, "%Y-%m-%d").date()]

    merge_from_to = merge_from_to.apply(pd.to_numeric, errors='ignore')

    merge_from_to = merge_from_to.sort_values(by=['region', 'date'])

    return merge_from_to


def save_merge_for_Poland():
    merge_data = get_merge_data_from_to()
    merge_data = merge_data[merge_data['region'] == 'POLSKA']
    merge_data.to_csv('results/merge_data_Poland.csv', index=False)
