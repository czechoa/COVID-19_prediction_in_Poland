import pandas as pd
import matplotlib.pyplot as plt
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
    plt.legend(loc='lower left')
    plt.show()
    fig.savefig(path)


def for_each_region_single_plot(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame):
    fig, ax = plt.subplots()

    date = data_merge_from_to_f['date'].unique()
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

def subplot_prediction_for_all_region(result_all_list: list, labels: list, data_merge_from_to_f: pd.DataFrame):
    regions = result_all_list[0]['region'].unique()
    z = 0
    while z < len(regions):
        print(z)
        plt.figure( figsize=(15, 15))
        for i in range(0, 4):
            if z + i >= len(regions): break
            plt.subplot(2,2,i+1)
            region = regions[z + i]
            date = data_merge_from_to_f['date'].unique()
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
