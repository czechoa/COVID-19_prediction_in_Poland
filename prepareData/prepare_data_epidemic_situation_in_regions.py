import pandas as pd


# train_data_path = 'data/COVID-19 w Polsce - Sytuacja epidemiczna w województwach.csv'
# data = pd.read_csv(train_data_path, header=1)
#
# # %%
# col_name = 'test'
# for col in data.columns:
#     if "Unnamed" in col:
#         data = data.rename(columns={col: col_name}, errors="raise")
#     else:
#         col_name = col


def prepare_data_epidemic_situation_in_regions(
        path='data/COVID-19 w Polsce - Sytuacja epidemiczna w województwach od 05.11 do 05.05.2021.csv'):
    train_data_path = path
    data = pd.read_csv(train_data_path, header=1)
    data = split_data_that_region_was_attribute(data)
    data = format_date(data)
    # data = data[data['region'] != 'POLSKA (SUMA)']
    data['region'] = data['region'].replace('POLSKA (SUMA)','POLSKA')
    data = data.drop(columns=['zmiana (d/d)'])
    # data = data.interpolate()
    data = drop_columns(data)
    return data


def split_data_that_region_was_attribute(data: pd.DataFrame):
    regions = data.columns.unique()
    regions = [s for s in regions if "Unnamed" not in s]
    data.columns = data.iloc[0]
    data = data[1:]
    n_regions = len(regions)
    data_region_as_attribute = pd.DataFrame()
    for i in range(1, n_regions):
        data_one_region = data.iloc[:, (1 + (i - 1) * 8):(9 + (i - 1) * 8)]
        data_one_region.insert(0, 'data', data.iloc[:, 0], True)
        data_one_region.insert(1, 'region', regions[i], True)
        data_region_as_attribute = data_region_as_attribute.append(data_one_region, ignore_index=True)
    return data_region_as_attribute


def format_date(data: pd.DataFrame):
    date = data.loc[:, 'data']
    date = [w.replace('.', '-') for w in date]
    new_formats = list()
    for day in date:
        if int(day[3:]) > 9:
            year = '2020-'
        else:
            year = '2021-'
        new_format = year + day[3:] + '-' + day[0:2]
        new_formats.append(new_format)
    data.loc[:, 'data'] = new_formats
    data = data.rename(columns={"data": "date"})
    data.iloc[:, -1] = reformed_percent(data.iloc[:, -1])
    data.iloc[:, -5] = reformed_percent(data.iloc[:, -5])
    return data


def reformed_percent(col: pd.Series):
    col = [float(w[:-1].replace(',', '.')) / 100 for w in col]
    return col


def drop_columns(data: pd.DataFrame):
    cols = data.columns.tolist()
    finale_cols = cols[:3]
    finale_cols.append(cols[-3])
    data = data[finale_cols]

    return data


def get_test_respiration(date='2021-04-11'):
    finale_data = prepare_data_epidemic_situation_in_regions(
        'data/COVID-19 w Polsce - Sytuacja epidemiczna w województwach od 05.11 do 05.05.2021.csv')
    finale_data = finale_data.drop(columns=[finale_data.columns[-2]])
    finale_day = finale_data[finale_data['date'] == date]
    return finale_day
# %%
# a = get_test_respiration(date='2021-04-11')
# data_region = prepare_data_epidemic_situation_in_regions()
