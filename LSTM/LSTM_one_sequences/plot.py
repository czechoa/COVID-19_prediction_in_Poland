import plotly.graph_objects as go

def get_org_data_from_region(region,split):
    date_train = region['date'].iloc[:int(region.shape[0] * split)]
    y_train_org = region['Engaged_respirator'].iloc[:int(region.shape[0] * split)]
    y_test_org = region['Engaged_respirator'].iloc[int(region.shape[0] * split):]
    date_test = region['date'].iloc[int(region.shape[0] * split):]
    return date_train, y_train_org, y_test_org, date_test

def get_org_data_from_region_make_plot(_region, _trainPredict, _testPredict, split):
    date_train, y_train_org, y_test_org, date_test = get_org_data_from_region(_region, split)
    trace1 = go.Scatter(
        x=date_train,
        y=y_train_org * 16,
        mode='lines',
        name='Data train'
    )
    trace2 = go.Scatter(
        # x=date_train[1:],
        x=date_train,
        y=_trainPredict[:, 0] * 16,
        mode='lines',
        name='Prediction train'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=_testPredict[:, 0] * 16,
        # y=testPredict*16,
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
