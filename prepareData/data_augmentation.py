import pandas as pd
import numpy as np


# from prepareData.merge_data_mobility_epidemic_situation import get_merge_data_from_to

def data_augmentation(merge_data_org: pd.DataFrame):
    merge_data = merge_data_org.append(return_region_as_weighted_average(merge_data_org), ignore_index=True)
    merge_data = merge_data.append(return_region_as_groupby_date(merge_data_org), ignore_index=True)
    merge_data = merge_data.append(Adding_Gaussian_Noise_to_avarage_region(merge_data), ignore_index=True)
    return merge_data


def return_region_as_weighted_average(merge_data: pd.DataFrame):
    data_merge_org = merge_data[merge_data['region'] != 'POLSKA']
    data_merge = data_merge_org.copy()
    data_merge.iloc[:, 3:] = (data_merge['population_%'].values * data_merge.iloc[:, 3:].values.T).T
    weighted_average_region: pd.DataFrame = data_merge.groupby(by=['date', 'day of the week']).sum().reset_index()
    weighted_average_region.insert(0, 'region', 'ŚŚ_weighted_average')
    # data_merge = data_merge_org.append(weighted_average_region, ignore_index=True)
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


def Adding_Gaussian_Noise_to_avarage_region(data_merge: pd.DataFrame, seed=2):
    np.random.seed(seed)
    average_region = data_merge[data_merge['region'] == 'ŚŚ_average']
    data_values_before = average_region.iloc[:, 3:].values
    l = (1 - np.random.normal(0, 0.03, data_values_before.shape))
    data_values = np.multiply(data_values_before, l)
    data = pd.DataFrame(columns=data_merge.columns[3:], data=data_values)
    data_dsc = average_region.iloc[:, [1, 2]].reset_index(drop=True)
    data = pd.concat([data_dsc, data], axis=1)
    data.insert(0, 'region', 'ŚŚ_Gaus_Noise')
    return data
# %%
# data_merge = get_merge_data_from_to()
# data_merge = data_augmentation(data_merge)

# %%
# average_region = data_merge[data_merge['region'] == 'ŚŚ_average']
# data_values_before =average_region.iloc[:, 3:].values
# l = (1 - np.random.normal(0, 0.03, data_values_before.shape))
# data_values =  np.multiply(data_values_before,l)
# data = pd.DataFrame(columns=data_merge.columns[3:], data=data_values)
# # data.iloc[:,1:3] = data_merge[data_merge['region'] == 'ŚŚ_average'].iloc[:,1:3]
# # data['region'] = 'ŚŚ_Gaus_1'
# # data.iloc[:,3:] = data_values
# data_dsc = average_region.iloc[:, [2, 3]].reset_index(drop=True)
# data = pd.concat([data_dsc, data], axis=1)
# data.insert(0, 'region', 'ŚŚ_Gaus')

# data['region'] = 'ŚŚ_Gaus_1'

# %%
# np.random.seed(1)
# data = get_merge_data_from_to()
# mean_std = list()
# new_region:pd.DataFrame = data.iloc[:,1:3]
# new_region.insert(0,'region','ŚŚ_gaus_noise')
# desc = data.groupby(by='date')
# for i in desc:
#     w = i.describe()
#     break
#
# # %%
# for column in  data.iloc[:,4:]:
#     desc = list(data[column].describe()[['mean','std']])
#     new_region[column] = np.random.normal(desc[0], desc[1], 1)
#     new_region.groupby(by='date').describe()
#
# # %%
#
# mu  =  data.mean()
# siqm = np.std(data)
#
# # %%
# w = np.random.normal(mu, siqm, 3)

# %%
# merge_data   = data_augmentation_without_Poland_as_sum(get_merge_data_from_to())
