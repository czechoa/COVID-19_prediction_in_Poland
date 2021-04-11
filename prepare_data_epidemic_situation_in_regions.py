import pandas as pd

train_data_path = 'data/COVID-19 w Polsce - Sytuacja epidemiczna w województwach.csv'
data = pd.read_csv(train_data_path, header=1)

# %%
col_name = 'test'
for col in data.columns:
    if "Unnamed" in col:
        data = data.rename(columns={col: col_name}, errors="raise")
    else:
        col_name = col


# %%
def prepare_data_epidemic_situation_in_regions():
    train_data_path = 'data/COVID-19 w Polsce - Sytuacja epidemiczna w województwach.csv'
    data = pd.read_csv(train_data_path, header=1)
    regions = data.columns.unique()
    regions = [s for s in regions if "Unnamed" not in s]
    print(regions)
    data.columns = data.iloc[0]
    data = data[1:]
    n_attribute = 8
    data_region_as_attribute = pd.DataFrame()
    for i in range(1, n_attribute):
        data_one_region = data.iloc[:, (1 + (i - 1) * 8):(9 + (i - 1) * 8)]
        data_one_region.insert(0, 'data', data.iloc[:, 0], True)
        data_one_region.insert(1, 'region', regions[i], True)
        data_region_as_attribute = data_region_as_attribute.append(data_one_region, ignore_index=True)
    return data_region_as_attribute


# %%
data = prepare_data_epidemic_situation_in_regions()
