import pandas as pd
import numpy as np
from prepare_data.merge.data_mobillity.dowload_unzip.download_unzip import get_unzip_data_regions_mobility


def prepare_date_mobility(data_path=None):
    # SUB_REGION_ 2 = 2
    if data_path is None:
        data_path = get_unzip_data_regions_mobility()
    i = list(range(8, 15))
    i.extend((2, 5))

    data: pd.DataFrame = pd.read_csv(data_path, usecols=i)
    data_Poland = data[data['sub_region_1'].isna()].copy()
    data_Poland['sub_region_1'] = 'Poland'

    data = data[data['iso_3166_2_code'].notna()]
    data = data.append(data_Poland)
    data = data.drop(columns='parks_percent_change_from_baseline')
    data = data.interpolate()

    return data


def day_of_the_week_for_all_regions(data: pd.DataFrame):
    number_of_region = len(data['sub_region_1'].unique())
    number_day = int(data.shape[0] / number_of_region)

    day_of_the_week = [(x % 7) / 6 for x in range(1, number_day + 1)]
    day_of_the_week_all = np.tile(day_of_the_week, number_of_region)
    data.insert(3, 'day of the week', day_of_the_week_all)
    return data


def merge_data_2020_2021(data_2020: pd.DataFrame, data_2021: pd.DataFrame):
    data_all = pd.concat([data_2020, data_2021])
    data_all = data_all.sort_values(by=[data_2021.columns[0], data_2021.columns[1]])
    return data_all


def to_percent_data_mobility(data):
    data.iloc[:, -5:] = data.iloc[:, -5:].div(100)
    return data


def get_prepared_data_mobility():
    train_data_path = 'data/data_input/2020_PL_Region_Mobility_Report.csv'
    data_2020 = prepare_date_mobility(train_data_path)

    # train_data_path_1 = 'data/data_input/2021-05-19_PL_Region_Mobility_Report.csv'
    # data_2021 = prepare_date_mobility(train_data_path_1)
    data_2021 = prepare_date_mobility()

    data_all = merge_data_2020_2021(data_2020, data_2021)

    data_all = day_of_the_week_for_all_regions(data_all)
    data_all = data_all.drop(columns='iso_3166_2_code')
    data_all = to_percent_data_mobility(data_all)

    return data_all
