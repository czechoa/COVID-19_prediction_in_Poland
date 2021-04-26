from prepare_data_mobility import *
from prepare_data_epidemic_situation_in_regions import prepare_data_epidemic_situation_in_regions
from simple_regresion import *


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
def reshape_data_merge_to_get_rows_with_n_days_history_and_target(data_merge: pd.DataFrame):
    print((data_merge.shape[0]))
    data_merge_with_history = pd.DataFrame()
    data_merge_with_history_test = pd.DataFrame()
    days_history = 14
    day_to_predict = 7
    n = days_history + day_to_predict
    number_of_days = len(data_merge.loc[:, 'date'].unique())
    while n < data_merge.shape[0]:
        if n % number_of_days == 0:  # new region
            data_merge_stack: pd.Series = data_merge.iloc[
                                          n - 1 - days_history:n - 1,
                                          3:].stack()
            data_merge_stack = pd.concat([data_merge.iloc[n - 1, :3], data_merge_stack]).reset_index(
                drop=True)
            data_merge_with_history_test = data_merge_with_history_test.append(data_merge_stack, ignore_index=True)

            n += days_history + day_to_predict
        else:
            data_merge_stack: pd.Series = data_merge.iloc[
                                          (n - days_history - day_to_predict):n - day_to_predict,
                                          3:].stack()
            data_merge_stack = pd.concat([data_merge.iloc[n, :3], data_merge_stack]).reset_index(
                drop=True)
            data_merge_stack['target'] = data_merge.iloc[n, -1]
            data_merge_with_history = data_merge_with_history.append(data_merge_stack, ignore_index=True)
            n += 1
    return data_merge_with_history, data_merge_with_history_test


# %%
def make_train_target(data_merge_with_history: pd.DataFrame):
    train = data_merge_with_history.iloc[:, :-1]
    train: pd.DataFrame = train.set_index([train.columns[0], train.columns[1]])
    train = train.astype(float)
    target = data_merge_with_history.iloc[:, -1]
    target = target.astype(float)
    return train, target


# %%

data_mobility, data_epidemic_situation_in_regions = get_data_mobility_data_epidemic_situation_in_regions()
data_mobility, data_epidemic_situation_in_regions = change_to_polish_name_regions_in_mobility_data(data_mobility,
                                                                                                   data_epidemic_situation_in_regions)
data_merge = merge_data_mobility_covid_19_situation(data_mobility, data_epidemic_situation_in_regions)

data_merge_with_history, data_merge_with_history_without_target = reshape_data_merge_to_get_rows_with_n_days_history_and_target(
    data_merge)

train, target = make_train_target(data_merge_with_history)

make_all(train, target, 'results/prediction_7')


# %%
def make_test(data_merge_with_history_without_target: pd.DataFrame):
    data_merge_with_history_without_target = data_merge_with_history_without_target.set_index(
        [data_merge_with_history_without_target.columns[0], data_merge_with_history_without_target.columns[1]])
    test = data_merge_with_history_without_target.astype(float)
    return test


# %%
test = make_test(data_merge_with_history_without_target)
submission = make_submission(test, 7)

submission = add_prediction_to_submission(test, submission, 7)
submission_to_cvs(submission, 'results/test_7')
