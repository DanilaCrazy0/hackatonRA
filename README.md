# Прогнозирование цен на электроэнергию на рынке на сутки вперед

Этот репозиторий содержит набор инструментов для анализа и прогнозирования цен на электроэнергию на основе данных с рынка на сутки вперед. Проект включает в себя парсинг данных, их обработку и различные методы прогнозирования временных рядов.

## Описание проекта

Проект направлен на сравнение различных методов прогнозирования цен на электроэнергию, включая:
- Классические методы машинного обучения (CatBoost)
- Временные ряды (SARIMA)
- Гибридные подходы (комбинация SARIMA/VAR с CatBoost)

Данные получаются через парсер с сайта рынка электроэнергии и содержат:
- Индексы равновесных цен
- Объемы покупки и продажи
- Минимальные и максимальные индексы цен

## Структура репозитория

1. **Парсинг данных**
   - `parser.py` - скрипт для сбора данных с сайта рынка электроэнергии

2. **Анализ данных**
   - `analysis.py` - визуализация данных и анализ временных рядов
   - `target_price.py` - обработка целевой переменной (интерполяция ночных значений)

3. **Методы прогнозирования**
   - `cross-val_cat_pred.py` - прогнозирование с CatBoost и кросс-валидацией
   - `lags_cat_pred.py` - CatBoost с добавлением лаговых признаков
   - `SARIMA_pred.py` - прогнозирование с использованием SARIMA
   - `SARIMA+cat_pred.py` - гибридный подход (SARIMA + CatBoost)
   - `VAR+catboost.py` - гибридный подход (VAR + CatBoost)

## Основные результаты

Проект демонстрирует сравнение различных подходов к прогнозированию цен на электроэнергию. В ходе работы были реализованы и протестированы:

1. **Классические ML-методы**:
   - CatBoost с кросс-валидацией
   - CatBoost с лаговыми признаками

2. **Методы временных рядов**:
   - SARIMA с учетом сезонности (24 часа)

3. **Гибридные подходы**:
   - Прогнозирование признаков через SARIMA/VAR с последующим использованием CatBoost
   - Комбинация статистических методов и машинного обучения

Для каждого метода рассчитываются метрики качества (RMSE, MAE) и строится визуализация результатов.

## Использование

1. Запустите `parser.py` для сбора актуальных данных
2. Используйте `analysis.py` для первичного анализа данных
3. Запускайте скрипты прогнозирования для сравнения методов

## Требования

- Python 3.7+
- Библиотеки: pandas, numpy, matplotlib, seaborn, scikit-learn, catboost, statsmodels, selenium
