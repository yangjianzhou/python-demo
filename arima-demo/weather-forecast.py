import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA as arima
import warnings
import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from pmdarima.arima import auto_arima
import sys


def read_temperature():
    data = pd.read_csv("/Users/jiyang12/Github/python-demo/arima-demo/data/weather-data.csv", usecols=[0, 1, 2, 3, 4])
    data['Date Time'] = pd.to_datetime(data['Date Time'])
    data.set_index("Date Time", inplace=True)
    data = data['Temperature']
    for i in range(len(data)):
        data[i] = round(5/9*(data[i] - 32))

    data = data.resample('30T').mean()
    data = data.fillna(data.bfill())
    data.plot(figsize=(15, 12))
    plt.show()
    return data


def adf_stability_test(data):
    x = data
    res = ts.adfuller(x)
    lb_res = lb_test(x, None, True)[1]

    tag = False
    for i in range(len(lb_res)):
        if lb_res[i] < 0.05:
            continue
        else:
            print('序列为白噪音')
            tag = True
            break

    if res[0] < res[4]['1%'] and res[0] < res[4]['5%'] and res[0] < res[4]['10%'] and res[1]<0.05 and not tag:
        print('平稳序列！非白噪音')
        return True
    else:
        return False


def stability_test(self):
    count = 0
    flag = False
    if not self.adf_stability_test(self.data):
        while not self.adf_stability_test(self.data):
            count+= 1
            self.data = self.data.dif(count)
            self.data = self.data.fillna(self.data.bfill())
            flag = True
    print('经过{}次差分，序列平稳'.format(count))
    return count, flag


def temp_param_optimization(self, data):
    paramBest = []
    warnings.filterwarnings("ignore")
    p = q = range(0, 3)		# 限制pq范围
    pdq = [(x[0], self.d, x[1]) for x in list(itertools.product(p, q))]	# 此处的d我们已经得到
    seasonal_pdq = [(x[0], self.d, x[1], 10) for x in list(itertools.product(p, q))]		# 此处的s是季节性参数，可根据数据指定，变化不大

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()
                paramBest.append([param, param_seasonal, results.aic])
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    # print('paramBest:', paramBest)
    minAic = sys.maxsize
    for i in np.arange(len(paramBest)):
        if paramBest[i][2] < minAic:
            minAic = paramBest[i][2]
    # print("minAic:", minAic)
    for j in np.arange(len(paramBest)):
        if paramBest[j][2] == minAic:
            return paramBest[j][0], paramBest[j][1]


def auto_parameters(self, data, s_num):
    kpss_diff = arima.ndiffs(data, alpha=0.05, test='kpss', max_d=s_num)
    adf_diff = arima.ndiffs(data, alpha=0.05, test='adf', max_d=s_num)
    d = max(kpss_diff, adf_diff)
    D = arima.nsdiffs(data, s_num)

    stepwise_model = auto_arima(data, start_p=1, start_q=1,
                                max_p=9, max_q=9, max_d=3, m=s_num,
                                seasonal=True, d=d, D=D, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    print("AIC: ", stepwise_model.aic())
    print(stepwise_model.order)		# (p,d,q)
    print(stepwise_model.seasonal_order)	# (P,D,Q,S)
    print(stepwise_model.summary())		# 详细模型
    return stepwise_model.order, stepwise_model.seasonal_order


def model_prediction(self, data, param, s_param, n_steps, flag):
    mod = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=s_param, enforce_stationarity=False, enforce_invertibility=False)
    results = mod.fit()

    pred_uc = results.get_forecast(steps=n_steps)		# n_steps可指定预测的步数(多少时间间隔)
    pred_ci = pred_uc.conf_int()

    if flag:  # 还原差分
        pred_res = pd.Series([data[0]], index=[data.index[0]]).append(pred_uc.predicted_mean).cumsum()
        print("预测结果(℃):  ", pred_res)
        return pred_res
    else:
        print("预测结果(℃):  ", pred_uc.predicted_mean)
        return pred_uc.predicted_mean


if __name__ == "__main__":
    data = read_temperature()
    adf_stability_test(data)