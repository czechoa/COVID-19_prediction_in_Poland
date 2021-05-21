
import matplotlib.pyplot as plt
from make_train_test_from_merge_data import *
from simple_regresion import *
from merge_data_mobility_epidemic_situation import get_merge_data_from_to
from prepare_data_epidemic_situation_in_regions import get_test_respiration
from datetime import datetime, timedelta


def next_day(date: str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")


# %%
last_day_train = '2021-03-01'

layers_n = 2

result_all = pd.DataFrame(columns=['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
                                   'prediction'])
result_all_err = pd.DataFrame()
period_of_time = 14

data_merge = get_merge_data_to_last_day(last_day_train)

# train_all = reshape_data_merge_to_get_train_with_two_week_history(data_merge, period_of_time)  # day is one day ahead
train_all = reshape_data_merge_to_get_train_period_of_time_history(data_merge, period_of_time)
test_to_predict = make_date_to_prediction(train_all)

day = last_day_train
for day_ahead_to_predict in range(1, 18):
    train, target = get_train_target(data_merge, train_all, period_of_time, day_ahead_to_predict)
    # test_to_predict = make_date_to_prediction(train_all)

    # train_sc, test_sc = standardScaler(train, test, input_scaler=MinMaxScaler())
    make_all(train, target)
    submission = make_submission(test_to_predict, day_ahead_to_predict)
    clear_model()

    # submission = add_prediction_to_submission(test_sc, submission, day_ahead_to_predict)

    day = next_day(day)
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

# %%
regions = result.loc[:, 'region'].unique()
data_merge_from_to = get_merge_data_from_to('2021-02-15','2021-04-04' )
zachodnie = data_merge_from_to.loc[data_merge_from_to['region'] == 'ZACHODNIOPOMORSKIE']
days_from_to = pd.to_datetime(zachodnie.loc[:, 'date'].values, format='%Y-%m-%d')

for i in range(len(regions)):
    fig, ax = plt.subplots()

    region_prd: pd.DataFrame = result_all.loc[result_all['region'] == regions[i]]
    region_merge = data_merge_from_to.loc[data_merge_from_to['region'] == regions[i]]

    y = region_merge.iloc[:, -1].astype(float)
    plt.plot(days_from_to, y)

    days = pd.to_datetime(region_prd.iloc[:, 0], format='%Y-%m-%d')

    x = days
    y = region_prd.loc[:, 'prediction'].astype(float).values
    plt.plot(x, y, label='prediction')
    # y = region_prd.iloc[:, -2].astype(float).values
    # plt.plot(x, y, label="reality")
    # x = list(days_from_to)
    # Define the d03ate format
    ax.set(xlabel="Date",
           ylabel="engaged respiration",
           title=regions[i]
           )
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
# %%
result_all_err = result_all_err.sort_values(by=['region','date'])