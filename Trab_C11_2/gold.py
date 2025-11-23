import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

df = pd.read_csv('gold.csv', delimiter = ',', index_col = 'Date' ,parse_dates=True)

df['China(CNY)'] = df['China(CNY)'].fillna(0)

df['China(CNY)'].plot(figsize = (8,6),
title = 'Preço de compra mensal do ouro na China',
xlabel = "Ano", 
ylabel = "Preço",
x_compat = True
)
plt.show()

decomposition = seasonal_decompose(df['China(CNY)'], model = 'additive', period = 12)

decomposition.plot()
plt.tight_layout()
plt.show()

model = ExponentialSmoothing(endog = df['China(CNY)'], trend = 'add', seasonal = 'add', seasonal_periods = 12).fit()

#prevendo 10 anos (120 meses)
predictions = model.forecast(steps=120)

df['China(CNY)']['31-01-2000':].plot(figsize=(8,6))

predictions.plot()
plt.show()

model2 = auto_arima(y = df['China(CNY)'], m =12)

freq = pd.infer_freq(df.index)
if freq is None:
    freq = 'M'
    
start = df.index[-1] + pd.tseries.frequencies.to_offset(freq)

#coloquei 36 meses para testar mas nao ta indo aaaaaaaaaaaaaaaaaaaaaaaaaaa
pred_index = pd.date_range(start=start, periods = 36, freq=freq)

predictions2 = pd.Series(arima_values, freq = freq)

df['China(CNY)']['31-01-2000':].plot(figsize=(8,6))

predictions2.plot()
plt.show()
