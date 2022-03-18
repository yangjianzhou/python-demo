import hvplot.pandas
import pandas as pd
import numpy as np
import panel as pn

def n1_sum(a, b, N, t, p):
    for n in range(N):
        n1_sum = np.sum(n1(a, b, n, t, p))
    return n1_sum
def n1(a, b, n, t, p):
    return a*np.cos(2*np.pi*n*t/p) + b*np.sin(2*np.pi*n*t/p)

a_widget = pn.widgets.DiscreteSlider(name='a', options=[-2, -1, 0, 1, 2 ], value=1)
b_widget = pn.widgets.DiscreteSlider(name='b', options=[-2, -1, 0, 1, 2 ], value=1)
N_widget = pn.widgets.DiscreteSlider(name='N', options=[1, 3, 5, 10], value=3)
p_widget = pn.widgets.DiscreteSlider(name='p', options=[7, 30, 90, 180, 365], value=7)

@pn.depends(a_widget, b_widget, N_widget, p_widget)
def plot(a_widget, b_widget, N_widget, p_widget):
    t = range(50)
    g = []
    for ti in t:
        g.append(n1_sum(a_widget, b_widget, N_widget, ti, p_widget))
    return pd.DataFrame({'time':t, 'value':g}).hvplot('time','value')

pn.Column(a_widget, b_widget, N_widget, p_widget, plot)