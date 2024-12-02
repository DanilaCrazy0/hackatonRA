import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', 10)

df = pd.read_csv('buyers_1_zone_value.csv')
data_start = df.iloc[0,0]
print(data_start)
data_end = df.iloc[df.shape[0]-1, 0]
df = df.drop(columns=['Дата', 'Час'])
df_h = df.index.to_numpy()
df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'] = (
    df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].str.replace(' ', ''))
df_eq_pr = df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].astype(float).to_numpy()
df_h = df_h[:505]
df_eq_pr = df_eq_pr[:505]
sma = np.zeros(len(df_eq_pr))
min_pr = np.zeros(len(df_eq_pr))
max_pr = np.zeros(len(df_eq_pr))

n = 24
sma[0:n] = 1/n * np.sum(df_eq_pr[:(n-1)])
for k in range(n, len(sma)):
    sma[k] = sma[k-1] - df_eq_pr[k-n]/n + df_eq_pr[k]/n

for k in range(0, len(sma), n):
    # sma[k:k+n] = np.mean(df_eq_pr[k:k+n])
    min_pr[k:k + n] = np.min(df_eq_pr[k:k + n])
    max_pr[k:k + n] = np.max(df_eq_pr[k:k + n])


def replace_zeros_with_min_nonzero(column):
    min_nonzero = column[column != 0].min()
    column[column == 0] = min_nonzero
    return column

for col in df.columns:
  df[col] = df[col].str.replace(' ', '').astype(float)

df['Минимальный индекс равновесной цены, руб./МВт.ч'] = replace_zeros_with_min_nonzero(df['Минимальный индекс равновесной цены, руб./МВт.ч'])

# plt.plot(df_h, df_eq_pr)
# plt.plot(df_h, sma, label='sma')
# plt.plot(df_h, min_pr, label='min')
# plt.plot(df_h, max_pr, label='max')
# plt.grid()

df = df.drop(columns=['Объем покупки на РСВ, МВт.ч', 'Объем продажи в обеспечение РД, МВт.ч'])
df = df.reset_index()

lags = 7 * 24

for col in df.columns[1:]:
    for lag in range(24, lags + 1, 24):
        df[f'{col}_lag_{lag // 24}'] = df[col].shift(lag)

df.dropna(inplace=True)
df = df.reset_index(drop=True)

train_data, test_data = train_test_split(df, shuffle=False, test_size=0.20)
x_train = train_data.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
y_train = train_data['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.']
x_test = test_data.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
y_test = test_data['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.']

model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=50)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(rmse)
print(mae*len(y_pred))

y_test = y_test.reset_index(drop=True)

plt.plot(y_test.index, y_test, label='test')
plt.plot(y_test.index, y_pred, label='pred')
plt.grid()
plt.legend()
plt.show()