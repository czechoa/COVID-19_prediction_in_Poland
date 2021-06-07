import pandas as pd

from make_prediction_one_month_ahead import make_prediction_one_mounth_ahead_for_train_all, make_plot_for_Poland
from merge_data_mobility_epidemic_situation import get_merge_data_from_to
from prepare_data_mobility import get_prepared_data_mobility

poland_resp_from_06_2020 = pd.read_csv("data/polska_respiration_from_11.06.2020.csv",delimiter = ';')
data_mobility:pd.DataFrame = get_prepared_data_mobility()
data_mobility_pl = data_mobility[data_mobility['sub_region_1'] == 'Poland']
data_merge_pl = pd.merge(data_mobility_pl,poland_resp_from_06_2020,on='date')
data_merge_pl: pd.DataFrame = data_merge_pl.drop(columns=['sub_region_1'])
data_merge_pl.insert(0, 'region', 'POLSKA')
data_merge_pl['date'] = pd.to_datetime(data_merge_pl[ 'date'], format='%Y-%m-%d').dt.date

data_merge_from_to = get_merge_data_from_to(str(data_merge_pl.iloc[-1,1]),'2021-03-20')
data_merge_from_to_pl = data_merge_from_to[data_merge_from_to['region'] == 'POLSKA']
data_merge_from_to_pl = data_merge_from_to_pl.iloc[1:]
data_merge_pl.rename(columns={data_merge_pl.columns[-2]: data_merge_from_to_pl.columns[-2], data_merge_pl.columns[-1]: data_merge_from_to_pl.columns[-1]},
                          inplace=True)
data_merge_from_to_pl = data_merge_from_to_pl.append(data_merge_pl,ignore_index=True)
data_merge_from_to_pl = data_merge_from_to_pl.sort_values(by='date')
# data_merge_pl['date'] = pd.to_datetime(data_merge_pl[:, 'date'], format='%Y-%m-%d').dt.date
# %%
result_all, result_all_err = make_prediction_one_mounth_ahead_for_train_all(data_merge_pl)
# %%
make_plot_for_Poland([result_all],['test'],title='data from 11.06,2020')
# %%
data_merge_pl.loc[:, 'region'].unique()
