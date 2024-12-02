import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Загрузка данных
df = pd.read_csv('buyers_1_zone_value.csv')
data_start = df.iloc[0, 0]
data_end = df.iloc[df.shape[0] - 1, 0]
df = df.drop(columns=['Дата', 'Час'])

# Обработка данных
df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'] = (
    df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].str.replace(' ', ''))
df_eq_pr = df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].astype(float).to_numpy()
df_h = df.index.to_numpy()

# Разделение данных на тренировочную и тестовую выборки
train_size = int(len(df_eq_pr) * 0.90)
train_data = df_eq_pr[:train_size]
test_data = df_eq_pr[train_size:]

# Обучение SARIMA
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)
model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Прогнозирование на тестовой выборке
forecast = model_fit.forecast(steps=len(test_data))

# Вычисление RMSE
rmse = np.sqrt(mean_squared_error(test_data, forecast))
print(f'RMSE: {rmse}')

# Визуализация результатов
plt.plot(df_h[train_size:], test_data, label='Actual')
plt.plot(df_h[train_size:], forecast, label='Forecast')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('SARIMA Forecast vs Actual')
plt.legend()
plt.show()