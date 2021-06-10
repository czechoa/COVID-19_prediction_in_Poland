import pandas as  pd
from sklearn.preprocessing import MinMaxScaler


def standardScaler(colum, input_scaler= MinMaxScaler()):
    if input_scaler is not None:
        std_column: pd.DataFrame = pd.DataFrame(index=colum.index[:], data=input_scaler.fit_transform(colum))

    return std_column

def preparing_data_area_population_regions():
    area_population = pd.read_csv('data/area_population_regions.csv')
    area_population.iloc[:, 1:] = area_population.iloc[:,1:].astype(float)
    tolat_population = float(area_population[area_population['region'] == 'POLSKA']['total'])
    # area_population['population_%'] = area_population['total'] / tolat_population
    # area_population['population_%'] = [ x/ tolat_population for x in area_population['total']]
    area_population['population_%'] = area_population['total'].div(tolat_population)
    area_population['region'] = [ x.upper() for x in area_population['region']]

    area_population.iloc[:, -2] = (area_population.iloc[:,-2]  - area_population.iloc[:,-2] .mean()) / area_population.iloc[:,-2] .std()
    # area_population.iloc[:,-2] = standardScaler(area_population.iloc[:,[-2]])
    area_population = area_population.iloc[:,[0,-2,-1]]
    return area_population
# %%
# area_population = preparing_data_area_population_regions()
