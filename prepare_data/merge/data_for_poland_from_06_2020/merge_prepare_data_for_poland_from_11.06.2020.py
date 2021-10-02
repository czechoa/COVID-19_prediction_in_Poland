import pandas as pd

from prepare_data.merge.data_mobillity.prepare_data_mobility import get_prepared_data_mobility
from prepare_data.merge.merge_data_mobility_epidemic_situation import get_merge_data_from_to
from prepare_data.augmentation.data_augmentation import adding_Gaussian_Noise_to_data_Poland
from prepare_data.test_train.make_train_test_from_merge_data import one_hot_encode


def merge_data_for_Poland_from_06_2020(last_day='2021-03-20'):
    poland_resp_from_06_2020 = pd.read_csv("data/data_input/polska_respiration_from_11.06.2020.csv", delimiter=';')
    data_mobility: pd.DataFrame = get_prepared_data_mobility()
    data_mobility_pl = data_mobility[data_mobility['sub_region_1'] == 'Poland']
    data_merge_pl = pd.merge(data_mobility_pl, poland_resp_from_06_2020, on='date')
    data_merge_pl: pd.DataFrame = data_merge_pl.drop(columns=['sub_region_1'])
    data_merge_pl.insert(0, 'region', 'POLSKA')
    data_merge_pl['date'] = pd.to_datetime(data_merge_pl['date'], format='%Y-%m-%d').dt.date

    data_merge_from_to = get_merge_data_from_to(str(data_merge_pl.iloc[-1, 1]),last_day)
    data_merge_from_to_pl = data_merge_from_to[data_merge_from_to['region'] == 'POLSKA']
    data_merge_from_to_pl = data_merge_from_to_pl.iloc[1:]
    data_merge_pl.rename(columns={data_merge_pl.columns[-2]: data_merge_from_to_pl.columns[-2],
                                  data_merge_pl.columns[-1]: data_merge_from_to_pl.columns[-1]},
                         inplace=True)
    data_merge_from_to_pl = data_merge_from_to_pl.append(data_merge_pl, ignore_index=True)
    data_merge_from_to_pl = data_merge_from_to_pl.sort_values(by='date')
    return data_merge_from_to_pl

def save_merge_data_for_Poland_from_06_2020_with_augumetation():
    data_Poland = merge_data_for_Poland_from_06_2020(last_day='2021-05-05')
    for i in range(10):
        data_Poland:pd.DataFrame = data_Poland.append(adding_Gaussian_Noise_to_data_Poland(data_Poland, i))
    data_Poland, number_desc = one_hot_encode(data_Poland, 'region', 3)
    data_Poland.to_csv('data/data_lstm/merge_data_Poland.csv',index=False)
