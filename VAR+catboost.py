import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import VAR
from catboost import CatBoostRegressor

# Загрузка данных
df = pd.read_csv('buyers_1_zone_value.csv')
pd.set_option('display.max_columns', None)
df = df.drop(columns=['Дата', 'Час'])

# Обработка данных
df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'] = (
    df['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].str.replace(' ', ''))

for col in df.columns:
    df[col] = df[col].str.replace(' ', '').astype(float)

df = df.drop(columns=['Объем покупки на РСВ, МВт.ч', 'Объем продажи в обеспечение РД, МВт.ч'])
df.dropna(inplace=True)

# Прогнозирование признаков с помощью VAR
def forecast_features_var(train_data, test_data, maxlags):
    # Обучение модели VAR
    model = VAR(train_data)
    model_fitted = model.fit(maxlags=maxlags)

    # Прогнозирование на тестовой выборке
    lag_order = model_fitted.k_ar
    forecast_input = train_data.values[-lag_order:]
    forecast = model_fitted.forecast(y=forecast_input, steps=len(test_data))

    # Преобразование прогноза в DataFrame
    forecast_df = pd.DataFrame(forecast, columns=train_data.columns)
    return forecast_df

input_time = 24 * 30 * 6
df = df.iloc[df.shape[0] - input_time : ].reset_index(drop=True)

prognoz_time = [24, 7*24, 30*24]

for elem in prognoz_time:
    test_size = elem / input_time
    # Разделение данных на тренировочную и тестовую выборки
    train_data, test_data = train_test_split(df, shuffle=False, test_size=test_size)

    # Прогнозирование с VAR
    forecasted_features = forecast_features_var(train_data, test_data, maxlags=24)

    # Обучение CatBoostRegressor на прогнозируемых признаках
    x_train = train_data.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
    y_train = train_data['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.']
    x_test = forecasted_features.drop(columns='Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.')
    y_test = test_data['Индекс равновесных цен на покупку электроэнергии, руб./МВт.ч.'].reset_index(drop=True)

    model = CatBoostRegressor(verbose=100)
    model.fit(x_train, y_train)

    # Предсказание таргета
    y_pred_test = model.predict(x_test)

    # Оценка метрик
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Истинные значения')
    plt.plot(y_test.index, y_pred_test, label='Прогноз')
    plt.title('Сравнение истинных значений и прогноза')
    plt.xlabel('Часы')
    plt.ylabel('Индекс равновесных цен')
    plt.legend()
    plt.grid()
    plt.show()

    # Построение корреляционной матрицы прогнозируемых признаков
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(forecasted_features.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title('Корреляционная матрица прогнозируемых признаков (VAR)')
    plt.show()
