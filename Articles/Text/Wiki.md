# Wiki

## Содержание
0. About
1. Как проверить ряд на стационарность
2. 

## About
В данном файле будет находится информация по применению тех или иных методов для подготовки и прогнозирования времянных данных.

### Как проверить ряд на стационарность?

1. Построить ряды скользящего среднего (rolling mean) и скльзящего стандартного отклонения (rolling std)
```python
window_size = 10
rolling_mean = model_data.Price.rolling(window_size).mean()
rolling_std = model_data.Price.rolling(window_size).mean()
plt.plot(model_data.Price, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
```
Если скользящее среднее и стандартное отклонение со временем возрастает, то ряд является нестационарным.

2. Провести тест Дики-Фуллера
```python
result = adfuller(model_data.Price)
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
```

Если p-value более 0.05, то мы считаем ряд нестационарным.

Если ряд стационарный, то следующий пункт можно пропустить.

### Как исправить нестационарность ряда?

Для исправления нестационарности ряда можно применить следующие методы:

1. Дифференцирование - это процесс вычитания каждого элемента ряда из предыдущего элемента. Это может помочь сделать ряд стационарным. Если ряд имеет тренд (линейное или нелинейное увеличение или уменьшение с течением времени), дифференцирование может убрать этот тренд.
```python
import pandas as pd
import numpy as np

# Предположим, у нас есть временной ряд `data`
# data = [здесь должны быть ваши данные временного ряда]

# Создаем DataFrame из временного ряда
df = pd.DataFrame(data, columns=['value'])

# Выполняем дифференцирование, вычитая каждый элемент ряда из предыдущего элемента
df['stationary_value'] = df['value'].diff()

# Удаляем первую строку, так как она будет содержать NaN (поскольку первого предыдущего элемента нет)
df = df.dropna()

# Теперь df['stationary_value'] содержит стационарный ряд после дифференцирования
print(df['stationary_value'])
```
2. Логарифмирование - это применение логарифма к каждому элементу ряда. Этот метод особенно полезен, когда ряд имеет экспоненциальный рост, что приводит к увеличению дисперсии с течением времени.
```python
import pandas as pd
import numpy as np

# Предположим, у нас есть временной ряд `data`
# data = [здесь должны быть ваши данные временного ряда]

# Создаем DataFrame из временного ряда
df = pd.DataFrame(data, columns=['value'])

# Применяем логарифмирование к каждому элементу ряда
df['stationary_value'] = np.log(df['value'])

# Теперь df['stationary_value'] содержит стационарный ряд после логарифмирования
print(df['stationary_value'])
```
3. Сглаживание - это метод устранения шумов и краткосрочных колебаний, что может сделать ряд более стационарным. Существуют различные методы сглаживания, такие как скользящее среднее, экспоненциальное сглаживание и т. д.
```python
import pandas as pd

# Предположим, у нас есть временной ряд `data`
# data = [здесь должны быть ваши данные временного ряда]

# Создаем DataFrame из временного ряда
df = pd.DataFrame(data, columns=['value'])

# Задаем окно скользящего среднего (например, 3 - среднее значение последних 3 элементов)
window_size = 3

# Применяем скользящее среднее к значению ряда
df['stationary_value'] = df['value'].rolling(window=window_size).mean()

# Удаляем строки с NaN, так как они будут в начале из-за окна скользящего среднего
df = df.dropna()

# Теперь df['stationary_value'] содержит стационарный ряд после применения скользящего среднего
print(df['stationary_value'])
```
4. Если ряд имеет ярко выраженную сезонность (циклические колебания с постоянным периодом), устранение сезонности может сделать ряд более стационарным. Это может быть достигнуто путем вычитания среднего значения для каждого периода.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Создадим симулированный временной ряд с сезонностью
# В данном примере, у нас будет сезонность с периодом 12 (12 месяцев)

# Генерируем случайные данные для сезонности
np.random.seed(0)
num_periods = 48
data = np.random.randint(10, 30, num_periods)

# Добавляем сезонные колебания с периодом 12
seasonality = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
data_with_seasonality = data + np.tile(seasonality, num_periods // 12)

# Создадим DataFrame с данными
date_range = pd.date_range(start='2020-01-01', periods=num_periods, freq='M')
df = pd.DataFrame({'Date': date_range, 'Data_with_Seasonality': data_with_seasonality})

# Отобразим исходные данные
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Data_with_Seasonality'], label='Исходные данные с сезонностью')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.title('Исходный временной ряд с сезонностью')
plt.legend()
plt.grid(True)
plt.show()

# Функция для применения скользящего среднего
def remove_seasonality(data, window_size):
    """
    Удаляет сезонность из временного ряда с помощью скользящего среднего.
    
    Параметры:
    data (array-like): Временной ряд с сезонностью.
    window_size (int): Размер окна для скользящего среднего.
    
    Возвращает:
    array-like: Временной ряд без сезонности.
    """
    rolling_mean = data.rolling(window=window_size, center=True).mean()
    data_without_seasonality = data - rolling_mean
    return data_without_seasonality

# Применяем скользящее среднее для удаления сезонности с окном размера 12 (для сезонности с периодом 12)
window_size = 12
df['Data_without_Seasonality'] = remove_seasonality(df['Data_with_Seasonality'], window_size)

# Отобразим временной ряд без сезонности
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Data_without_Seasonality'], label='Временной ряд без сезонности')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.title('Временной ряд без сезонности')
plt.legend()
plt.grid(True)
plt.show()
```
5. Разделение ряда на его составляющие, такие как тренд, сезонность и остатки, может помочь в анализе и представлении ряда в более стационарном виде.
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Предположим, что у нас есть временной ряд с именем 'time_series'
# Пример случайного сезонного ряда:
np.random.seed(42)
time_series = pd.Series(np.random.randint(0, 100, size=100), name='value')

# Предполагаем, что сезонность у нас равна 10
seasonality_period = 10

# Добавим сезонность к ряду (симулируем сезонные данные)
time_series = time_series + np.sin(np.arange(len(time_series)) * (2 * np.pi / seasonality_period)) * 10

# Применим аддитивную декомпозицию для разделения ряда на тренд, сезонность и остатки
decomposition = sm.tsa.seasonal_decompose(time_series, model='additive', period=seasonality_period)

# Получим остатки (ряд без сезонности)
residuals = time_series - decomposition.seasonal

# Теперь 'residuals' - это ряд без сезонности, который стал более стационарным

# Можно вывести графики для визуализации декомпозиции
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(time_series, label='Original')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(residuals, label='Residuals')
plt.legend(loc='upper left')
plt.show()
```
6. Преобразование Бокса-Кокса - это параметрическое преобразование, которое может стабилизировать дисперсию ряда и сделать его более стационарным.
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Пример нестационарного временного ряда (замените его своим рядом)
# Можно использовать, например, библиотеку Pandas для чтения временного ряда из файла
time_series = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Проверка на наличие отрицательных значений и сдвиг, если нужно
min_value = np.min(time_series)
if min_value < 0:
    time_series = time_series - min_value + 1

# Применение преобразования Бокса-Кокса
transformed_data, lambda_value = stats.boxcox(time_series)

# Вывод значения параметра lambda, которое было использовано в преобразовании
print("Значение параметра lambda:", lambda_value)

# Вывод исходного и преобразованного ряда
print("Исходный ряд:", time_series)
print("Преобразованный ряд:", transformed_data)

# Визуализация исходного и преобразованного ряда
plt.figure(figsize=(10, 5))
plt.plot(time_series, label='Исходный ряд')
plt.plot(transformed_data, label='Преобразованный ряд')
plt.legend()
plt.xlabel('Временные точки')
plt.ylabel('Значения')
plt.title('Преобразование Бокса-Кокса')
plt.show()
```
7. Если ряд содержит как тренд, так и сезонность, можно использовать методы регрессии для их удаления и получения стационарного остатка.
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Пример временного ряда
data = pd.Series([i + np.sin(i/10) + np.random.randn(1)[0] for i in range(100)])
# Создадим индекс времени (предполагая, что данные у нас упорядочены по времени)
time_index = pd.date_range(start='2023-01-01', periods=len(data), freq='D')

# Добавим столбец индекса времени в DataFrame с данными
df = pd.DataFrame({'data': data.values}, index=time_index)

# Создадим столбец дня недели (понедельник - 0, воскресенье - 6)
df['day_of_week'] = df.index.dayofweek

# Создадим столбец с номерами дней (начиная с 1)
df['day_number'] = df.index.day

# Добавим столбец с номерами месяцев (начиная с 1)
df['month_number'] = df.index.month

# Создадим дамми-переменные для каждого месяца и дня недели
dummy_months = pd.get_dummies(df['month_number'], prefix='month')
dummy_days = pd.get_dummies(df['day_of_week'], prefix='day')

# Объединим DataFrame с данными и дамми-переменными
df = pd.concat([df, dummy_months, dummy_days], axis=1)

# Определим матрицу X и вектор y для регрессии
X = df[['day_number', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
        'month_12', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6']]
y = df['data']

# Выполним регрессию
model = sm.OLS(y, sm.add_constant(X))
results = model.fit()

# Получим остатки
residuals = results.resid

# Отобразим результаты
print(results.summary())

# Теперь `residuals` содержит стационарный ряд без тренда и сезонности.
```
