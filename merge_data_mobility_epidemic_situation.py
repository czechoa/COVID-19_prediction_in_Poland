from prepare_data_mobility import *
from prepare_data_epidemic_situation_in_regions import prepare_data_epidemic_situation_in_regions

# %%
data_mobility = get_prepared_data_mobility()
data_epidemic_situation_in_regions = prepare_data_epidemic_situation_in_regions()
# %%
data_epidemic_situation_in_regions.sort_values(by='region')
region = data_epidemic_situation_in_regions.loc[:, 'region'].unique()
region = list(region)
# %%
data_mobility.sort_values(by='sub_region_1')
region_mobility = data_mobility.loc[:, 'sub_region_1'].unique()
region_mobility = list(region_mobility)

# %%
region_df = pd.DataFrame()
region_df['mobility'] = region_mobility
region_df['region_epidemic_situation'] = region
# %%
myorder = [14, 1, 5, 0, 2, 3, 6, 7, 8, 9, 10, 11, 13, 15, 4, 12]
region = [region[i] for i in myorder]
region_df['region_epidemic_situation'] = region
# %%
number_date = data_mobility.shape[0] / len(region_mobility)
region_repeat = np.repeat(region, number_date)
# %%
data_mobility.insert(1, 'region', region_repeat)
# %%
data_merge = data_mobility.merge(data_epidemic_situation_in_regions, how='inner', on=['region', 'date'])
# %%
data_merge: pd.DataFrame = data_merge.drop(columns=[data_merge.columns[0]])
# %%

number_of_region = len(region)
number_of_days = len(data_merge.loc[:, 'date'].unique())

# %%
data_merge_with_history = pd.DataFrame()
days_history = 14
day_to_predict = 7
n = days_history + day_to_predict
while n < data_merge.shape[0]:
    if n % number_of_days == 0:
        n += days_history + day_to_predict
    else:
        data_merge_stack: pd.Series = data_merge.iloc[
                                      (n - days_history - day_to_predict):n - day_to_predict,
                                      3:].stack()
        data_merge_stack = pd.concat([data_merge.iloc[n, :3], data_merge_stack]).reset_index(
            drop=True)
        data_merge_stack['target'] = data_merge.iloc[n,-1]
        data_merge_with_history = data_merge_with_history.append(data_merge_stack, ignore_index=True)
        n += 1
# %%
train = data_merge_with_history.iloc[:, :-1]
train: pd.DataFrame = train.set_index([train.columns[0], train.columns[1]])
train = train.astype(float)
target = data_merge_with_history.iloc[:, -1]
target = target.astype(float)
# %%
from simple_regresion import *

make_all(train, target, 'results/prediction_7')
# %%
