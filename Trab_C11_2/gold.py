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
ylabel = "Preço (Yuan)",
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

df['China(CNY)']['31-01-2000':].plot(figsize=(8,6), title = 'Preço de compra mensal do ouro na China',
xlabel = "Ano", 
ylabel = "Preço (Yuan)")

predictions.plot()
plt.show()

model2 = auto_arima(df['China(CNY)'], m=12, seasonal=True, trace=True)

freq = pd.infer_freq(df.index)
if freq is None:
    freq = 'M'

start = df.index[-1] + pd.tseries.frequencies.to_offset(freq)

#prevendo 10 anos (120 meses)
pred_index = pd.date_range(start = start, periods = 120, freq = freq)

arima_values = model2.predict(n_periods = 120)

predictions2 = pd.Series(arima_values, index = pred_index)

df['China(CNY)']['31-01-2000':].plot(figsize=(8,6), title = 'Preço de compra mensal do ouro na China',
xlabel = "Ano", 
ylabel = "Preço (Yuan)")

predictions2.plot()
plt.show()

def MAPE(y_true, y_pred):
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Mape Holf Winters
hw_fitted = model.fittedvalues

real_hw = df['China(CNY)'].loc[hw_fitted.index]

mape_hw = MAPE(real_hw, hw_fitted)

print("MAPE Holt-Winters:", mape_hw)

#Mape Arima

arima_fitted = model2.predict_in_sample()

arima_fitted = pd.Series(arima_fitted, index=df.index)

real_arima = df['China(CNY)']

mape_arima = MAPE(real_arima, arima_fitted)

print("MAPE ARIMA:", mape_arima)
