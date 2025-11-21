import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('gold.csv', delimiter = ',', index_col = 'Date' ,parse_dates=True)

df['China(CNY)'] = df['China(CNY)'].fillna(0)

df['China(CNY)'].plot(figsize = (8,6),
title = 'Preço anual do ouro por ano',
xlabel = "Ano", 
ylabel = "Preço",
x_compat = True
)
plt.show()

decomposition = seasonal_decompose(df['China(CNY)'], model = 'additive', period = 12)

decomposition.plot()
plt.tight_layout()
plt.show()