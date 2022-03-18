import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    discFile = "/Users/jiyang12/Github/python-demo/arima-demo/data/arima-test.xlsx"
    forecastNum = 5
    data = pd.read_excel(discFile ,index_col=u'date')
    plt.rcParams['font.sans-serif']=['SimHei']

    plt.rcParams['axes.unicode_minus']=False
    data.plot()
    plt.show()


if __name__ == "__main__":
    load_data()