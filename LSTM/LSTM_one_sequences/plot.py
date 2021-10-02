import plotly.graph_objects as go


class Plot:
    def __init__(self, region, trainPredict, testPredict, split):
        self.split = split
        self.trainPredict = trainPredict
        self.region = region
        self.testPredict = testPredict

    def _get_org_data_from_region(self):
        n_rows = self.region.shape[0]
        last_index_train = int(n_rows * self.split)
        date_train = self.region['date'].iloc[:last_index_train]
        y_train_org = self.region['Engaged_respirator'].iloc[:last_index_train]
        y_test_org = self.region['Engaged_respirator'].iloc[last_index_train:]
        date_test = self.region['date'].iloc[last_index_train:]
        return date_train, y_train_org, y_test_org, date_test

    def get_org_data_from_region_make_plot(self):
        date_train, y_train_org, y_test_org, date_test = self._get_org_data_from_region()
        trace1 = go.Scatter(
            x=date_train,
            y=y_train_org * 16,
            mode='lines',
            name='Data train'
        )
        trace2 = go.Scatter(
            x=date_train,
            y=self.trainPredict[:, 0] * 16,
            mode='lines',
            name='Prediction train'
        )
        trace3 = go.Scatter(
            x=date_test,
            y=self.testPredict[:, 0] * 16,
            mode='lines',
            name='Prediction future'
        )

        trace5 = go.Scatter(
            x=date_test,
            y=y_test_org * 16,
            mode='lines',
            name='Ground true'
        )

        layout = go.Layout(
            title="Poland covid-19",
            xaxis={'title': "Date"},
            yaxis={'title': "Engaged respirator"}
        )

        fig = go.Figure(data=[trace1, trace2, trace3, trace5], layout=layout)

        fig.show()
