import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Загрузка данных
df = pd.read_csv('buyers_1_zone_value.csv')
pd.set_option('display.max_columns', None)
data_start = df.iloc[0, 0]
data_end = df.iloc[df.shape[0] - 1, 0]
df = df.drop(columns=['Дата', 'Час'])


# Обработка данных
df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'] = (
    df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].str.replace(' ', ''))


for col in df.columns:
    df[col] = df[col].str.replace(' ', '').astype(float)

df = df.drop(columns=['Объем покупки на РСВ, МВт.ч', 'Объем продажи в обеспечение РД, МВт.ч'])
df.dropna(inplace=True)
df = df.reset_index(drop=True)

print(df)
# Разделение данных на тренировочную и тестовую выборки
train_data, test_data = train_test_split(df, shuffle=False, test_size=0.05)

# Разделение тренировочной выборки на тренировочную1 и тестовую1
train_data1, test_data1 = train_test_split(train_data, shuffle=True, test_size=0.20)

# Обучение SARIMA для прогнозирования признаков
def forecast_features(train_data, test_data, order, seasonal_order):
    forecasts = {}
    for col in train_data.columns:
        if col != 'Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.':
            model = SARIMAX(train_data[col], order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(test_data))
            forecasts[col] = forecast
            print(col)
    return pd.DataFrame(forecasts)

# Параметры SARIMA
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)

# Прогнозирование признаков на тестовой выборке
forecasted_features = forecast_features(train_data, test_data, order, seasonal_order)

# Обучение CatBoostRegressor на тренировочной1 выборке
x_train1 = train_data1.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
y_train1 = train_data1['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.']
x_test1 = test_data1.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
y_test1 = test_data1['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.']

model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=50)
model.fit(x_train1, y_train1)

# Оценка модели на тестовой1 выборке
y_pred1 = model.predict(x_test1)
rmse1 = np.sqrt(mean_squared_error(y_test1, y_pred1))
mae1 = mean_absolute_error(y_test1, y_pred1)
print(f'RMSE на тестовой1 выборке: {rmse1}')
print(f'MAE на тестовой1 выборке: {mae1}')

# Дообучение модели на всей тренировочной выборке
x_train = train_data.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
y_train = train_data['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.']
model.fit(x_train, y_train, init_model=model)

# Предсказание таргета на основе прогнозируемых признаков
y_pred_test = model.predict(forecasted_features)

# Визуализация результатов
y_test = test_data['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].reset_index(drop=True)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print(f'RMSE на тестовой1 выборке: {rmse}')
print(f'MAE на тестовой1 выборке: {mae * len(y_test)}')

plt.plot(y_test.index, y_test, label='test')
plt.plot(y_test.index, y_pred_test, label='pred')
plt.grid()
plt.legend()
plt.show()