import pandas as pd
import numpy as np
# %%
from preparingData import get_combined_data, get_cols_with_no_nans

train_data_path = 'data/2020_PL_Region_Mobility_Report.csv'
i = list(range(8, 15))
i.append(2)
print(i)
# %%
data = pd.read_csv(train_data_path, usecols=i)
print(data.columns.values)
# %%
# data = data.drop(['metro_area', 'census_fips_code'], axis=1)
# print(data.shape)
# print(data.columns.values)
# %%
print("before dropna ", data.shape)
data = data.dropna()
print("after dropna ", data.shape)
# %%

# num_cols = get_cols_with_no_nans(data, 'num')
# cat_cols = get_cols_with_no_nans(data, 'no_num')
# print ('Number of numerical columns with no nan values :',len(num_cols))
# print ('Number of nun-numerical columns with no nan values :',len(cat_cols))
# print(combined.shape)
# combined = combined[num_cols + cat_cols]
# %%
print("before mean by regions ", data.shape)
data = data.groupby(by=['sub_region_1', 'date']).mean().reset_index()
print("before mean by regions ", data.shape)
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
# %% add colum day_of_the_week (  2020-02-15 was a Tuesday so start from 2)
day_of_the_week = [x % 7 + 1 for x in range(1, size[0] + 1)]
print(len(day_of_the_week))
print(len(data_day_week['sub_region_1'].unique()))
day_of_the_week_all = np.repeat(day_of_the_week, len(data_day_week['sub_region_1'].unique()))
# %%
data_day_week_region_1: pd.DataFrame = data_day_week[data_day_week['sub_region_1'] == 'Greater Poland Voivodeship']
data_day_week_region_1 = data_day_week_region_1.sort_values(by='date')
# %%
len(day_of_the_week_all)
data_day_week.shape[0]
