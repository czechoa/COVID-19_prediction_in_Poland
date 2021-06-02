import matplotlib.pyplot as plt
from make_train_test_from_merge_data import *
from simple_regresion import *
from merge_data_mobility_epidemic_situation import get_merge_data_from_to, get_merge_data_to_last_day, get_merge_data
from prepare_data_epidemic_situation_in_regions import get_test_respiration
from datetime import datetime, timedelta


#  3 dniowa srednia  i moze oneHotCode
#  dane dla polski i suma predykcji (  na jednym wykresie )
# wyslac wykresy
#   SSTM - LTFTM
#
def next_day(date: str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")


def make_prediction_one_mounth_ahead_for_train_all(data_merge, period_of_time=21, last_day_train='2021-03-20'):
    # last_day_train = '2021-03-20'
    # period_of_time = 21

    # data_merge = data_merge[data_merge["region"] != 'POLSKA']
    # train_all = reshape_data_merge_to_get_train_period_of_time_history(data_merge, period_of_time)
    train_all = reshape_data_merge_to_get_train_period_of_time_history_1(data_merge, period_of_time)

    test_to_predict = make_date_to_prediction(train_all)

    result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
                                       'prediction'])
    result_all_err = pd.DataFrame()
    day = next_day(last_day_train)
    for day_ahead_to_predict in range(1, 31):
        train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)

        # train, test_to_predict = standardScaler(train, test_to_predict, input_scaler=MinMaxScaler())
        # train = avarage_train_from_n_days(train, 3)

        make_all(train, target)
        submission = make_submission(test_to_predict, day_ahead_to_predict)
        clear_model()

        # submission = add_prediction_to_submission(test_sc, submission, day_ahead_to_predict)

        submission = submission.reset_index()
        test_ahead: pd.DataFrame = get_test_respiration(date=day)

        submission.rename(columns={submission.columns[0]: test_ahead.columns[1], submission.columns[2]: 'prediction'},
                          inplace=True)
        result = pd.merge(test_ahead, submission.drop(columns=submission.columns[1]), on=test_ahead.columns[1])
        result_err = result.iloc[:, :2]
        result_err['subtract'] = result.iloc[:, -2].astype(float) - result.iloc[:, -1].astype(float)
        result_err['relative error in %'] = abs(result_err.loc[:, 'subtract'] / result.iloc[:, -1].astype(float)) * 100
        result_all = result_all.append(result, ignore_index=True)
        result_all_err = result_all_err.append(result_err, ignore_index=True)
        day = next_day(day)

    #
    # norm_2 = np.linalg.norm(result_err['relative error in %'], ord=2)
    # print(norm_2)
    result_all = result_all.sort_values(by=['region', 'date'])
    result_all_err = result_all_err.sort_values(by=['region', 'date'])

    return result_all, result_all_err


# def plot_prediction_to_Poland(result_all_f, data_merge_from_to_f):
#     fig, ax = plt.subplots()
#
#     # first_region = data_merge_from_to.loc[data_merge_from_to['region'] == 'ZACHODNIOPOMORSKIE']
#     # days_from_to = pd.to_datetime(first_region.loc[:, 'date'].values, format='%Y-%m-%d')
#
#     date = data_merge_from_to['date'].unique()
#     days_from_to = pd.to_datetime(date, format='%Y-%m-%d')
#
#     region_merge = data_merge_from_to_f.loc[data_merge_from_to_f['region'] == "POLSKA"]
#     y = region_merge.iloc[:, -1].astype(float)
#     plt.plot(days_from_to, y, label="reality")
#
#     polska_prd: pd.DataFrame = result_all_f.loc[result_all_f['region'] == 'POLSKA']
#     days = pd.to_datetime(polska_prd.iloc[:, 0], format='%Y-%m-%d')
#
#     x = days
#     y = polska_prd.loc[:, 'prediction'].astype(float).values
#     plt.plot(x, y, label='prediction')
#
#     ax.set(xlabel="Date",
#            ylabel="engaged respiration",
#            title='POLSKA'
#            )
#     plt.gcf().autofmt_xdate()
#     plt.grid()
#     plt.legend(loc='lower left')
#     plt.show()
#     # fig.savefig("results/Poland predition when you learn from sum")


def plot_prediction_to_Poland_from_results(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame
                                           ,path = "results/Poland prediction engaged respiration when learn from all set"
                                           ,title='Poland prediction engaged respiration'):
    fig, ax = plt.subplots()

    date = data_merge_from_to['date'].unique()
    days_from_to = pd.to_datetime(date, format='%Y-%m-%d')

    region_merge = data_merge_from_to_f.loc[data_merge_from_to_f['region'] == "POLSKA"]
    y = region_merge.iloc[:, -1].astype(float)
    plt.plot(days_from_to, y, label="reality")

    for result_all_f, label in zip(result_all_list, labels):
        polska_prd: pd.DataFrame = result_all_f.loc[result_all_f['region'] == 'POLSKA']
        days = pd.to_datetime(polska_prd.iloc[:, 0], format='%Y-%m-%d')
        y = polska_prd['prediction'].astype(float).values
        plt.plot(days, y, label=label)

    ax.set(xlabel="Date",
           ylabel="Engaged respiration",
           title=title
           )
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    fig.savefig(path)


def for_each_region_single_plot(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame):
    fig, ax = plt.subplots()

    # first_region = data_merge_from_to.loc[data_merge_from_to['region'] == 'ZACHODNIOPOMORSKIE']
    # days_from_to = pd.to_datetime(first_region.loc[:, 'date'].values, format='%Y-%m-%d')

    date = data_merge_from_to['date'].unique()
    days_from_to = pd.to_datetime(date, format='%Y-%m-%d')
    for region in data_merge_from_to_f['region'].unique():
        region_merge = data_merge_from_to_f.loc[data_merge_from_to_f['region'] == region]
        y = region_merge.iloc[:, -1].astype(float)
        plt.plot(days_from_to, y, label="reality")

        for result_all_f, label in zip(result_all_list, labels):
            region_prd: pd.DataFrame = result_all_f.loc[result_all_f['region'] == region]
            days = pd.to_datetime(region_prd.iloc[:, 0], format='%Y-%m-%d')
            y = region_prd['prediction'].astype(float).values
            plt.plot(days, y, label=label)

        # plt.scatter(days_from_to[0], y)
        # plt.annotate("Point 1", (1, 4))

        ax.set(xlabel="Date",
               ylabel="Engaged respiration",
               )
        plt.title(region + ' prediction engaged respiration')
        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.legend(loc='lower left')
        plt.show()


def save_list_results(list_results, path = 'results/all_prediction_for_region.csv'):
    all_prediction = pd.DataFrame()
    for result_all_it, i in zip(list_results.copy(), list([1, 3, 7])):
        result_all_it.insert(2, 'avarage from n days back', str(i))
        all_prediction = all_prediction.append(result_all_it, ignore_index=True)
        result_all_it.drop(columns='avarage from n days back', inplace=True)
    all_prediction.to_csv(path, index=False)

def from_all_prection_to_list_results(all_prediction:pd.DataFrame):
    list_results = list()
    for avarage_from_n in all_prediction['avarage from n days back'].unique():
        result = all_prediction[all_prediction['avarage from n days back'] == avarage_from_n]
        result = result.drop(columns='avarage from n days back')
        list_results.append(result)
    return list_results




def plot_prediction_for_each_16_region(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame):
    regions = result_all_list[0]['region'].unique()
    z = 0
    while z < len(regions):
        print(z)
        plt.figure( figsize=(15, 15))
        for i in range(0, 4):
            if z + i >= len(regions): break
            plt.subplot(2,2,i+1)
            region = regions[z + i]
            date = data_merge_from_to['date'].unique()
            days_from_to = pd.to_datetime(date, format='%Y-%m-%d')
            region_merge = data_merge_from_to_f.loc[data_merge_from_to_f['region'] == region]
            y = region_merge.iloc[:, -1].astype(float)
            plt.plot(days_from_to, y, label="reality")

            for result_all_f, label in zip(result_all_list, labels):
                region_prd: pd.DataFrame = result_all_f.loc[result_all_f['region'] == region]
                days = pd.to_datetime(region_prd.iloc[:, 0], format='%Y-%m-%d')
                y = region_prd['prediction'].astype(float).values
                plt.plot(days, y, label=label)

            # plt.scatter(days_from_to[0], y)
            # plt.annotate("Point 1", (1, 4))

            plt.xlabel(xlabel="Date")
            plt.ylabel(ylabel="Engaged respiration")
            plt.title(region)

            plt.gcf().autofmt_xdate()
            plt.grid()
            plt.legend(loc='lower left')
        z += i + 1
        plt.savefig('results/region_prediction_' + str(z),bbox_inches='tight')
        plt.show()

# %%
last_day_train = '2021-03-20'
period_of_time = 21
data_merge_org = get_merge_data_from_to(last_day=last_day_train)
data_merge_org = data_merge_org[data_merge_org["region"] != 'POLSKA']
# %%
polska_grub:pd.DataFrame =  data_merge_org.groupby(by= 'date').mean().reset_index()
polska_grub.insert(0,'region','POLSKA')
data_merge_org = data_merge_org.append(polska_grub,ignore_index=True)
data_merge_org = data_merge_org.sort_values(by= ['region','date'])
# %%
list_results = list()
labels = list()
for i in [1, 3, 7]:
    data_merge = avarage_merge_data_from_n_days(data_merge_org.copy(), i)
    result_all, result_all_err = make_prediction_one_mounth_ahead_for_train_all(data_merge, period_of_time,
                                                                                last_day_train)
    label = 'prediction from averaged ' + str(i) + ' days back'
    list_results.append(result_all)
    labels.append(label)

data_merge_from_to = get_merge_data_from_to('2021-03-01', '2021-05-01')
# %%
for result_all_it in list_results:
    result_all_it['prediction'] = result_all_it['prediction'] * 16

# %%
title = 'Poland engaged respiration, learned by mean Poland'
plot_prediction_to_Poland_from_results(list_results, labels, data_merge_from_to,path= 'results/' + title, title=title)
# %%
title = 'Poland engaged respiration, learned by mean Poland'
save_list_results(list_results,path='results/'+ title)
# %%plot_prediction_to_Poland_from_results(list_results, labels, data_merge_from_to)

data_merge_from_to = get_merge_data_from_to('2021-03-01', '2021-05-01')
all_prediction_for_regions = pd.read_csv('results/all_prediction_for_region.csv')
list_results = from_all_prection_to_list_results(all_prediction_for_regions)
labels  =list()
for i in [1, 3, 7]:
    label = 'prediction from averaged ' + str(i) + ' days back'
    labels.append(label)
# %%
plot_prediction_for_each_16_region(list_results, labels, data_merge_from_to)

# %%
for_each_region_single_plot(list_results, labels, data_merge_from_to)
# %%
plot_prediction_to_Poland_from_results(list_results, labels, data_merge_from_to)
# %%
