import pandas as pd

from prepareData.merge_data_mobility_epidemic_situation import get_merge_data_from_to


def data_augmentation_without_Poland_as_sum(merge_data_org: pd.DataFrame):
    merge_data = merge_data_org.append( return_region_as_weighted_average(merge_data_org), ignore_index=True)
    merge_data = merge_data.append( return_region_as_groupby_date(merge_data_org), ignore_index=True)
    return merge_data



def return_region_as_weighted_average(merge_data: pd.DataFrame):
    data_merge_org = merge_data[merge_data['region'] != 'POLSKA']
    # merge_all.insert([3,4],merge_all.columns[[-2,-1]].values,merge_all.iloc[:,[-2,-1]])
    # data_merge['avareage for poland'] = data_merge['population_%'].values * data_merge.iloc[:,-1].values.T
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
# %%
# merge_data   = data_augmentation_without_Poland_as_sum(get_merge_data_from_to())
