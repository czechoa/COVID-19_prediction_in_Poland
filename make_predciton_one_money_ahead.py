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

def make_prediction_one_mounth_ahead_for_train_all(data_merge, train_all, period_of_time, last_day_train):

    result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
                                       'prediction'])
    result_all_err = pd.DataFrame()
    day = next_day(last_day_train)
    for day_ahead_to_predict in range(1,31):
        train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)

        # train, test_to_predict = standardScaler(train, test_to_predict, input_scaler=MinMaxScaler())
        # train = avarage_train_from_n_days(train, 3)

        make_all(train, target)
        submission = make_submission(test_to_predict, day_ahead_to_predict)
        clear_model()

        day = next_day(day)
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
    #
    # norm_2 = np.linalg.norm(result_err['relative error in %'], ord=2)
    # print(norm_2)
    result_all = result_all.sort_values(by=['region', 'date'])
    result_all_err = result_all_err.sort_values(by=['region', 'date'])

    return result_all, result_all_err

def plot_prediction_to_Poland(result_all_f, data_merge_from_to_f):
    fig, ax = plt.subplots()

    region_prd: pd.DataFrame = result_all_f.loc[result_all_f['region'] == 'POLSKA']
    region_merge = data_merge_from_to_f.loc[data_merge_from_to_f['region'] == "POLSKA"]

    y = region_merge.iloc[:, -1].astype(float)

    zachodnie = data_merge_from_to.loc[data_merge_from_to['region'] == 'ZACHODNIOPOMORSKIE']
    days_from_to = pd.to_datetime(zachodnie.loc[:, 'date'].values, format='%Y-%m-%d')

    plt.plot(days_from_to, y, label="reality")

    days = pd.to_datetime(region_prd.iloc[:, 0], format='%Y-%m-%d')

    x = days
    y = region_prd.loc[:, 'prediction'].astype(float).values
    plt.plot(x, y, label='prediction')
    ax.set(xlabel="Date",
           ylabel="engaged respiration",
           title= 'POLSKA'
           )
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # fig.savefig("results/Poland predition when you learn from sum")

# %%
last_day_train = '2021-03-20'
period_of_time = 21
data_merge = get_merge_data_from_to(last_day = last_day_train)
# data_merge = data_merge[data_merge["region"] != 'POLSKA']
# train_all = reshape_data_merge_to_get_train_period_of_time_history(data_merge, period_of_time)
train_all = reshape_data_merge_to_get_train_period_of_time_history_1(data_merge, period_of_time)

test_to_predict = make_date_to_prediction(train_all)
# %%

result_all, result_all_err = make_prediction_one_mounth_ahead_for_train_all(data_merge,train_all,period_of_time,last_day_train)
data_merge_from_to = get_merge_data_from_to('2021-03-01', '2021-05-01')

plot_prediction_to_Poland(result_all,data_merge_from_to)

# %%
def plot_prediction_for_ech_region():
    fig, ax = plt.subplots(8,2 , figsize=(60, 30))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = ax.ravel()
    for i in range(len(regions)):


        region_prd: pd.DataFrame = result_all.loc[result_all['region'] == regions[i]]
        region_merge = data_merge_from_to.loc[data_merge_from_to['region'] == regions[i]]

        y = region_merge.iloc[:, -1].astype(float)
        axs[i].plot(days_from_to, y, label="reality")

        days = pd.to_datetime(region_prd.iloc[:, 0], format='%Y-%m-%d')

        x = days
        y = region_prd.loc[:, 'prediction'].astype(float).values
        axs[i].plot(x, y, label='prediction')
        # y = region_prd.iloc[:, -2].astype(float).values
        # plt.plot(x, y, label="reality")
        # x = list(days_from_to)
        # Define the d03ate format
        axs[i].set(xlabel="Date",
               ylabel="engaged respiration",
               title=regions[i]
               )
        # axs[i].gcf().autofmt_xdate()
        axs[i].grid()
        axs[i].legend(loc='lower left')
    plt.show()
    fig.savefig('results/test_plot')

