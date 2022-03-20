import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA as arima
import warnings
import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from pmdarima.arima import auto_arima


def forecast_sales():
    discFile = "/Users/jiyang12/Github/python-demo/arima-demo/data/arima-test.xlsx"
    forecastNum = 5
    data = pd.read_excel(discFile ,index_col=u'date')
    plt.rcParams['font.sans-serif']=['SimHei']

    plt.rcParams['axes.unicode_minus']=False
    data.plot()

    plot_acf(data).show()
    print(u'original adf result :', ADF(data[u'sales']))

    D_data = data.diff().dropna()
    D_data.columns = [u'sales diff']
    D_data.plot()

    plot_acf(D_data).show()

    plt.show()
    #plot_pacf(D_data).show()

    pmax = int(len(D_data)/10)
    qmax = int(len(D_data)/10)

    bic_matrix = []
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(arima(data,(p,1,q)).fit().bic)
            except:
                tmp.append(None)
            bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    p,q = bic_matrix.stack().idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' %(p, q))

    model =arima(data, (p,1,q)).fit()
    model.summary2()
    model.forecast(5)


if __name__ == "__main__":
    forecast_sales()