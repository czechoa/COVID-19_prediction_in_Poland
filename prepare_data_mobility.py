import pandas as pd
import numpy as np

# %%


train_data_path = 'data/2020_PL_Region_Mobility_Report.csv'
i = list(range(8, 15))
i.append(2)
i.append(5)

print(i)
# %%
data = pd.read_csv(train_data_path, usecols=i)
# %%
print(data.columns.values)
# %%
data = data[data['iso_3166_2_code'].notna()]
print(data['iso_3166_2_code'].unique())
# %%
data_day_week = data.copy()
# %% check number of day is the same of all regions (
size = 0
sum = 0
regions_with_no_all_day = list()
for region in data_day_week['sub_region_1'].unique():
    size = data_day_week[data_day_week['sub_region_1'] == region].shape
    # if size == new_size or size == 0:
    #     size = new_size
    #     sum += size[0]
    # else:
    #     print("error")
    #     # Exception("number of columns is different so probably number of day for regions is different")
    tmp = region + " " + str(size[0])
    regions_with_no_all_day.append(tmp)
print(data_day_week.shape)
# %%
data_isna = data[data.isna().any(axis=1)]
data = data.drop(columns='parks_percent_change_from_baseline')
# %% without_park because many nan value
data_isna = data[data.isna().any(axis=1)]
# %%
data = data.interpolate()
# %%
data_isna = data[data.isna().any(axis=1)]
# %% add colum day_of_the_week (  2020-02-15 was a Tuesday so start from 2)

day_of_the_week = [x % 7 + 1 for x in range(1, size[0] + 1)]
print(len(day_of_the_week))
print(len(data_day_week['sub_region_1'].unique()))
day_of_the_week_all = np.tile(day_of_the_week, len(data_day_week['sub_region_1'].unique()))
data['day of the week'] = day_of_the_week_all

# %%
data['day of the week'] = day_of_the_week_all


# %% all in two function
def prepare_date_mobility(data_path):
    i = list(range(8, 15))
    i.append(2)
    i.append(5)
    print(data_path)
    data = pd.read_csv(data_path, usecols=i)
    data = data[data['iso_3166_2_code'].notna()]
    data = data.drop(columns='parks_percent_change_from_baseline')
    data = data.interpolate()
    # data = day_of_the_week_for_all_regions(data)
    return data


def day_of_the_week_for_all_regions(Data: pd.DataFrame):
    day_of_the_week = [x % 7 + 1 for x in range(1, size[0] + 1)]
    day_of_the_week_all = np.tile(day_of_the_week, len(data_day_week['sub_region_1'].unique()))
    data['day of the week'] = day_of_the_week_all
    return data


def merge_data_2020_2021(data_2020: pd.DataFrame, data_2021: pd.DataFrame):
    data_all = pd.concat([data_2020, data_2021])
    data_all = data_all.sort_values(by=[data_2021.columns[0], data_2021.columns[1]])
    return data_all


# %%
train_data_path = 'data/2020_PL_Region_Mobility_Report.csv'
data_2020 = prepare_date_mobility(train_data_path)
# %%
train_data_path_1 = 'data/2021_PL_Region_Mobility_Report.csv'
data_2021 = prepare_date_mobility(train_data_path_1)
# %%
data_all  = merge_data_2020_2021(data_2020,data_2021)
