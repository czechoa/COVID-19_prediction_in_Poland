import pandas as pd
import numpy as np


def data_augmentation(merge_data_org: pd.DataFrame, number_of_gaussian_noise_regions = 1):
    merge_data = merge_data_org.append(return_region_as_weighted_average(merge_data_org), ignore_index=True)
    merge_data = merge_data.append(return_region_as_groupby_date(merge_data_org), ignore_index=True)
    for i in range(number_of_gaussian_noise_regions):
        merge_data = merge_data.append(adding_Gaussian_Noise_to_averaged_region(merge_data,seed=i), ignore_index=True)

    return merge_data


def return_region_as_weighted_average(merge_data: pd.DataFrame, first_dsc_attribute = 3):
    data_merge_org = merge_data[merge_data['region'] != 'POLSKA']
    data_merge = data_merge_org.copy()

    data_merge.iloc[:, first_dsc_attribute:] = (data_merge['population_%'].values * data_merge.iloc[:, first_dsc_attribute:].values.T).T
    weighted_average_region: pd.DataFrame = data_merge.groupby(by=['date', 'day of the week']).sum().reset_index()
    weighted_average_region.insert(0, 'region', 'ŚŚ_weighted_average')

    return weighted_average_region


def return_region_as_groupby_date(data_merge_f: pd.DataFrame):
    data_merge_org = data_merge_f[data_merge_f["region"] != 'POLSKA']
    data_merge = data_merge_org.copy()

    average_region: pd.DataFrame = data_merge.groupby(by=['date', 'day of the week']).mean().reset_index()
    average_region.insert(0, 'region', 'ŚŚ_average')

    return average_region


def add_region_Poland_as_mean_groupby_date(data_merge_f: pd.DataFrame):
    data_merge_f = data_merge_f[data_merge_f["region"] != 'POLSKA']

    poland_grub: pd.DataFrame = data_merge_f.groupby(by='date').mean().reset_index()

    poland_grub.insert(0, 'region', 'POLSKA')
    data_merge_f = data_merge_f.append(poland_grub, ignore_index=True)

    data_merge_f = data_merge_f.sort_values(by=['region', 'date'])

    return data_merge_f


def adding_Gaussian_Noise_to_averaged_region(data_merge: pd.DataFrame, seed=2):
    np.random.seed(seed)

    average_region = data_merge[data_merge['region'] == 'ŚŚ_average']
    data_values_before = average_region.iloc[:, 3:].values

    gauss_noise_parameters = (1 - np.random.normal(0, 0.03, data_values_before.shape))
    data_values = np.multiply(data_values_before, gauss_noise_parameters)

    data = pd.DataFrame(columns=data_merge.columns[3:], data=data_values)
    data_dsc = average_region.iloc[:, [1, 2]].reset_index(drop=True)
    data = pd.concat([data_dsc, data], axis=1)
    data.insert(0, 'region', 'ŚŚ_Gaus_Noise_seed_'+str(seed))

    return data

def adding_Gaussian_Noise_to_data_Poland(data_poland: pd.DataFrame, seed=2):
    np.random.seed(seed)

    average_region = data_poland[data_poland['region'] == 'POLSKA']
    data_values_before = average_region.iloc[:, 3:].values

    gauss_noise_parameters = (1 - np.random.normal(0, 0.03, data_values_before.shape))
    data_values = np.multiply(data_values_before, gauss_noise_parameters)

    data = pd.DataFrame(columns=data_poland.columns[3:], data=data_values)
    data_dsc = average_region.iloc[:, [1, 2]].reset_index(drop=True)
    data = pd.concat([data_dsc, data], axis=1)
    data.insert(0, 'region', 'ŚŚ_Gaus_Noise_seed_'+str(seed))

    return data
