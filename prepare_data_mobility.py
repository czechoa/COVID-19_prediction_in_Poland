import pandas as pd
from preparingData import get_combined_data, get_cols_with_no_nans

train_data_path = 'data/2020_PL_Region_Mobility_Report.csv'
i = range(2, 15)
data = pd.read_csv(train_data_path, usecols=i)=True
data = data.drop(['metro_area', 'census_fips_code'], axis=1)
print(data.shape)
print(data.columns.values)
print(data.iloc[:, 0].unique())

# num_cols = get_cols_with_no_nans(data, 'num')
# cat_cols = get_cols_with_no_nans(data, 'no_num')
# print ('Number of numerical columns with no nan values :',len(num_cols))
# print ('Number of nun-numerical columns with no nan values :',len(cat_cols))
# print(combined.shape)
# combined = combined[num_cols + cat_cols]
# %%
print(data.shape)
data = data.groupby(by=['sub_region_1', 'date']).mean().reset_index()
# %%
data['residential_percent_change_from_baseline']

