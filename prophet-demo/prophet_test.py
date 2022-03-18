import prophet
import pandas
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
import numpy as np


def plot_figure():
    print("prophet version : %s" % prophet.__version__)
    df = pandas.read_csv("/Users/jiyang12/Github/python-demo/data/monthly-car-sales.cvs", header=0)
    df.columns = ['ds', 'y']
    df['ds'] = pandas.to_datetime(df['ds'])

    pyplot.plot(df['ds'], df['y'])
    pyplot.show()


def forecast_future():
    print("prophet version : %s" % prophet.__version__)
    df = pandas.read_csv("/Users/jiyang12/Github/python-demo/data/monthly-car-sales.cvs", header=0)
    print(df.shape)
    print(df.head())
    df.columns = ['ds', 'y']
    df['ds'] = pandas.to_datetime(df['ds'])
    model = prophet.Prophet()
    model.fit(df)

    future = list()
    for i in range(1, 13):
        date = '1968-%02d' % i
        future.append([date])
    future = pandas.DataFrame(future)
    future.columns = ['ds']
    future['ds'] = pandas.to_datetime(future['ds'])
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(n= 13))
    model.plot(forecast)
    #df.plot()

    pyplot.show()


def forecast_pass():

    df = pandas.read_csv("/Users/jiyang12/Github/python-demo/data/monthly-car-sales.cvs", header=0)
    df.columns = ['ds', 'y']
    df['ds'] = pandas.to_datetime(df['ds'])
    #train = df.drop(df.index[-12:])
    model = prophet.Prophet(interval_width=0.8)
    #model.add_country_holidays(country_name='CN')
    model.fit(df)

    future = list()
    for i in df['ds'][-108:]:
        future.append(i)
    future = pandas.DataFrame(future)
    future.columns = ['ds']
#    for i in range(1, 13):
#        date = '1968-%02d' % i
#        future.append([date])
#    future = pandas.DataFrame(future)
 #   future.columns = ['ds']
    future['ds'] = pandas.to_datetime(future['ds'])
    forecast = model.predict(future)
    y_true = df['y'][-108:].values
    y_pred = forecast['yhat'].values
    print("forecast " ,y_pred)
    y_pred_lower = forecast['yhat_lower'].values
    #y_pred_upper = forecast['yhat_upper'].values
    mae = mean_absolute_error(y_true, y_pred)
    print('MAE: %.3f' % mae)
    #pyplot.plot(y_true, label= 'actual')
    #pyplot.plot(y_pred, label= 'predicted')
    #pyplot.plot(y_pred_lower, label= 'y_pred_lower')
    #pyplot.plot(y_pred_upper, label= 'y_pred_upper')
    #pyplot.legend()

    #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(n= 13))
    model.plot(forecast)

    #df.plot()
    #pyplot.show()
    #model.plot_components(forecast)

    pyplot.show()


def mape_x(y_true, y_predict):
    return np.average(np.abs(y_true - y_predict)/y_true * 100)


def tuning_model():
    df = pandas.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv", header=0)
    df.columns = ['ds', 'y']
    df['ds'] = pandas.to_datetime(df['ds'])
    model = prophet.Prophet(growth='linear',changepoint_prior_scale=0.1, seasonality_mode='multiplicative', yearly_seasonality=20, seasonality_prior_scale=20, interval_width=0.5)
    model.fit(df)

    future = list()
    for i in df['ds'][-108:]:
        future.append(i)
    future = pandas.DataFrame(future)
    future.columns = ['ds']
    future['ds'] = pandas.to_datetime(future['ds'])
    forecast = model.predict(future)
    y_true = df['y'][-108:].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    print('MAE: %.3f' % mae)
    mape = mape_x(y_true, y_pred)
    print('mape: %.3f' % mape)
    model.plot(forecast)
    model.plot_components(forecast)

    pyplot.show()


def test_model():
    df = pandas.read_csv("/Users/jiyang12/Github/python-demo/data/test-data.csv", header=0)
    df.columns = ['ds', 'y']
    df['ds'] = pandas.to_datetime(df['ds'])
    model = prophet.Prophet()
    model.add_country_holidays('KR')
    model.fit(df)

    future = list()
    for i in df['ds'][-813:]:
        future.append(i)
    future = pandas.DataFrame(future)
    future.columns = ['ds']
    future['ds'] = pandas.to_datetime(future['ds'])
    forecast = model.predict(future)

    y_true = df['y'][-813:].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    print('MAE: %.3f' % mae)
    mape = mape_x(y_true, y_pred)
    print('mape: %.3f' % mape)
    fig=model.plot(forecast)

    model.plot_components(forecast)
    pyplot.show()


if __name__ == '__main__':
    print("prophet version : %s" % prophet.__version__)
    forecast_pass()
    #forecast_future()
    #test_model()
    #tuning_model()
    #plot_figure()

