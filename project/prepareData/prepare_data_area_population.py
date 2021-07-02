import pandas as pd

def preparing_data_area_population_regions():
    area_population = pd.read_csv('data/area_population_regions.csv')
    area_population.iloc[:, 1:] = area_population.iloc[:, 1:].astype(float)
    total_population = float(area_population[area_population['region'] == 'POLSKA']['total'])
    area_population['population_%'] = area_population['total'].div(total_population)

    area_population['region'] = [x.upper() for x in area_population['region']]

    area_population.iloc[:, -2] = (area_population.iloc[:, -2] - area_population.iloc[:,
                                                                 -2].mean()) / area_population.iloc[:, -2].std()

    area_population = area_population.iloc[:, [0, -2, -1]]
    return area_population


