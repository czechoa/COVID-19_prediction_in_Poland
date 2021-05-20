# %%
from make_train_test_from_merge_data import get_all_train_test_target
from simple_regresion import *
from prepare_data_epidemic_situation_in_regions import get_test_respiration

train, test, target = get_all_train_test_target(period_of_time=21)
train_sc, test_sc = standardScaler(train, test, input_scaler=MinMaxScaler())
# %%
from datetime import datetime, timedelta
def next_day(date :str):
    date = datetime.strptime(date, "%Y-%m-%d")
    modified_date = date + timedelta(days=1)
    return datetime.strftime(modified_date, "%Y-%m-%d")

# %%
layers_n = 2
last_day_train = '2021-04-04'
day = last_day_train
result_all = pd.DataFrame(columns= ['date', 'region', 'Liczba zajętych respiratorów (stan ciężki)',
       'prediction'])
result_all_err = pd.DataFrame()
for day_ahead_to_predict in range(1,30):

    make_all(train_sc, target, layers_n)
    submission = make_submission(test_sc, day_ahead_to_predict)
    # print("--- %s seconds ---" % (time.time() - start_time))
    clear_model()
    day = next_day(day)
    submission = submission.reset_index()
    test_ahead: pd.DataFrame = get_test_respiration(date = day)
    submission.rename(columns={submission.columns[0]: test_ahead.columns[1], submission.columns[2]: 'prediction'},
                      inplace=True)
    result = pd.merge(test_ahead, submission.drop(columns=submission.columns[1]), on=test_ahead.columns[1])
    result_err = result.iloc[:, :2]
    result_err['subtract'] = result.iloc[:, -2].astype(float) - result.iloc[:, -1].astype(float)
    result_err['relative error in %'] = abs(result_err.loc[:, 'subtract'] / result.iloc[:, -1].astype(float)) * 100
    result_all = result_all.append(result, ignore_index=True)
    result_all_err = result_all_err.append(result_err, ignore_index=True)
    # norm_2 = np.linalg.norm(result_err['relative error in %'], ord=2)
    # print(norm_2)


# %%
regions = result.loc[:,'region'].unique()
mazowsze_prd:pd.DataFrame =  result_all.loc[ result_all['region'] == regions[6]]
# %%
import matplotlib.pyplot as plt
# %%
days = pd.to_datetime(mazowsze_prd.iloc[:,0], format='%Y-%m-%d')
plt.plot_date(days,mazowsze_prd.loc[:,'prediction'].astype(float).values)
plt.show()
# %%
x = days
y = mazowsze_prd.loc[:,'prediction'].astype(float).values
plt.plot(x,y)
y = mazowsze_prd.iloc[:,-2].astype(float).values
plt.plot(x,y)
plt.show()
# %%
x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)

plt.show()
# %%
days = pd.to_datetime(mazowsze_prd.iloc[:,0])
# %%
fig, ax = plt.subplots()
x = days
y = mazowsze_prd.loc[:,'prediction'].astype(float).values
plt.plot(x,y,label= 'prediction')
y = mazowsze_prd.iloc[:,-2].astype(float).values
plt.plot(x,y, label="reality")
# Define the date format
ax.set(xlabel="Date",
       ylabel="engaged respiration",
       title="Mazowsze"

       )
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend( loc='lower left')

plt.show()
