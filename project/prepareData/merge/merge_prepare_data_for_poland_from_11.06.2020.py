import pandas as pd
from project.prediction.make_prediction_n_days_ahead import make_prediction_n_days_ahead, make_plot_for_Poland
from merge_data_mobility_epidemic_situation import get_merge_data_from_to
from project.prepareData.prepare_data_mobility import get_prepared_data_mobility


def merge_data_for_Poland_from_06_2020(last_day='2021-03-20'):
    poland_resp_from_06_2020 = pd.read_csv("data/polska_respiration_from_11.06.2020.csv", delimiter=';')
    data_mobility: pd.DataFrame = get_prepared_data_mobility()
    data_mobility_pl = data_mobility[data_mobility['sub_region_1'] == 'Poland']
    data_merge_pl = pd.merge(data_mobility_pl, poland_resp_from_06_2020, on='date')
    data_merge_pl: pd.DataFrame = data_merge_pl.drop(columns=['sub_region_1'])
    data_merge_pl.insert(0, 'region', 'POLSKA')
    data_merge_pl['date'] = pd.to_datetime(data_merge_pl['date'], format='%Y-%m-%d').dt.date

    data_merge_from_to = get_merge_data_from_to(str(data_merge_pl.iloc[-1, 1]), last_day)
    data_merge_from_to_pl = data_merge_from_to[data_merge_from_to['region'] == 'POLSKA']
    data_merge_from_to_pl = data_merge_from_to_pl.iloc[1:]
    data_merge_pl.rename(columns={data_merge_pl.columns[-2]: data_merge_from_to_pl.columns[-2],
                                  data_merge_pl.columns[-1]: data_merge_from_to_pl.columns[-1]},
                         inplace=True)
    data_merge_from_to_pl = data_merge_from_to_pl.append(data_merge_pl, ignore_index=True)
    data_merge_from_to_pl = data_merge_from_to_pl.sort_values(by='date')
    return data_merge_from_to_pl


def poland_prediction_average_of_10_measurements():
    data_merge_pl = merge_data_for_Poland_from_06_2020()
    data_merge_to_2021_05 = merge_data_for_Poland_from_06_2020(last_day='2021-05-01')
    sum_result_all = pd.DataFrame()
    for i in range(0, 10):
        result_all, result_all_err = make_prediction_n_days_ahead(data_merge_pl, day_ahead=30)
        if i == 0:
            sum_result_all = result_all
        else:
            sum_result_all = sum_result_all + result_all
    sum_result_all.iloc[:, -1] = sum_result_all.iloc[:, -1].div(10)
    sum_result_all['date'] = result_all['date']
    sum_result_all['region'] = result_all['region']
    make_plot_for_Poland([sum_result_all], ['prediction'], title='Poland prediction average of 10 measurements',
                         data_merge_from_to=data_merge_to_2021_05, save=True)
