import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('buyers_1_zone_value.csv')
pd.set_option('display.max_columns', None)
data_start = df.iloc[0, 0]
data_end = df.iloc[df.shape[0] - 1, 0]
df = df.drop(columns=['Дата', 'Час'])


# Обработка данных
for col in df.columns:
  df[col] = df[col].str.replace(' ', '').astype(float)


df_h = df.index.to_numpy()
df_eq_pr = df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].astype(float).to_numpy()
period = 30*24
df_h = df_h[:period]
df_eq_pr = df_eq_pr[:period]
sma = np.zeros(len(df_eq_pr))
min_pr = np.zeros(len(df_eq_pr))
max_pr = np.zeros(len(df_eq_pr))

n = 24
sma[0:n] = 1 / n * np.sum(df_eq_pr[:(n - 1)])
for k in range(n, len(sma)):
    sma[k] = sma[k - 1] - df_eq_pr[k - n] / n + df_eq_pr[k] / n

for k in range(0, len(sma), n):
    min_pr[k:k + n] = np.min(df_eq_pr[k:k + n])
    max_pr[k:k + n] = np.max(df_eq_pr[k:k + n])

plt.plot(df_h, df_eq_pr)
plt.plot(df_h, sma, label='sma')
plt.plot(df_h, min_pr, label='min')
plt.plot(df_h, max_pr, label='max')
plt.grid()
plt.show()

min_ind = df['Минимальный индекс равновесной цены, руб./МВт.ч'].to_numpy()[:period]
max_ind = df['Максимальный индекс равновесной цены, руб./МВт.ч'].to_numpy()[:period]

plt.plot(df_h, df_eq_pr, label='ex')
plt.plot(df_h, max_ind, label='max')
plt.plot(df_h, min_ind, label='min')
plt.grid()
plt.legend()
plt.plot()

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.show()