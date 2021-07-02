import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_to_Poland_from_results(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame
                                           ,
                                           path="results/Poland prediction engaged respiration when learn from all set"
                                           , title='Poland prediction engaged respiration'):
    fig, ax = plt.subplots()

    date = data_merge_from_to_f['date'].unique()
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
    plt.legend(loc='upper left')
    plt.show()
    fig.savefig(path)



def subplot_prediction_for_all_region(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame):
    regions = result_all_list[0]['region'].unique()
    z = 0
    date = data_merge_from_to_f['date'].unique()

    while z < len(regions):
        print(z)
        plt.figure(figsize=(15, 15))
        for i in range(0, 4):
            if z + i >= len(regions): break
            plt.subplot(2, 2, i + 1)
            region = regions[z + i]
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
        plt.savefig('results/region_prediction_' + str(z), bbox_inches='tight')
        plt.show()


def subplot_relative_error_for_all_region(result_all):
    regions = result_all['region'].unique()
    z = 0
    date = result_all['date'].unique()
    days_from_to = pd.to_datetime(date, format='%Y-%m-%d')
    plt.figure(figsize=(15, 15))
    while z < len(regions):
        plt.subplot(3, 2, int(z / 4) + 1)
        # avarage_relative_error = np.zeros(len(days_from_to))
        for i in range(0, 4):

            if z + i >= len(regions): break

            region = regions[z + i]

            region_result = result_all[result_all['region'] == region]
            y = region_result.iloc[:, -1].astype(float)
            plt.plot(days_from_to, y, label=str(region))
            # avarage_relative_error = avarage_relative_error.__add__(y.values)
            # plt.scatter(days_from_to[0], y)
            # plt.annotate("Point 1", (1, 4))

        # avarage_relative_error = avarage_relative_error/4
        # plt.plot(days_from_to, avarage_relative_error,'k--',label='test')
        plt.xlabel(xlabel="Date")
        plt.ylabel(ylabel="Relative_error_engaged_respiration")
        plt.title("Regions_prediction_relative_error_" + str(z + 4))

        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.legend(loc='upper left')
        z += 4
    plt.savefig('results/regions_prediction_relative_error_1', bbox_inches='tight')
    plt.show()


def plot_averaged_relative_error_for_all_region(result_all):
    regions = result_all['region'].unique()
    z = 0
    date = result_all['date'].unique()
    days_from_to = pd.to_datetime(date, format='%Y-%m-%d')
    plt.figure(figsize=(15, 15))
    avarage_relative_error = np.zeros(len(days_from_to))
    for i in range(0, len(regions)):
        region = regions[i]

        region_result = result_all[result_all['region'] == region]
        y = region_result.iloc[:, -1].astype(float)
        avarage_relative_error = avarage_relative_error.__add__(y.values)
        # plt.scatter(days_from_to[0], y)
        # plt.annotate("Point 1", (1, 4))

    avarage_relative_error = avarage_relative_error / len(regions)

    plt.plot(days_from_to, avarage_relative_error, 'k--')
    plt.xlabel(xlabel="Date")
    plt.ylabel(ylabel="Relative_error")
    plt.title("Regions_prediction_averaged_relative_error")

    plt.gcf().autofmt_xdate()
    plt.grid()
    # plt.legend(loc='upper left')
    plt.savefig('results/regions_prediction_relative_error_averaged', bbox_inches='tight')
    plt.show()


def plot_relative_error_for_Polska(result_poland: pd.DataFrame):
    date = result_poland['date'].unique()
    days_from_to = pd.to_datetime(date, format='%Y-%m-%d')
    plt.figure(figsize=(15, 15))

    y = result_poland.iloc[:, -1]
    plt.plot(days_from_to, y, 'k--')
    plt.xlabel(xlabel="Date")
    plt.ylabel(ylabel="Relative_error")
    plt.title("Poland_prediction_relative_error")

    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.savefig('results/Poland_prediction_relative_error', bbox_inches='tight')
    plt.show()
