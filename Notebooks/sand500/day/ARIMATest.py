import sys
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

# Функция для построения модели ARIMA и получения прогноза
def build_arima_model_and_forecast(data, p, d, q, window_size = 200, forecast_days = 90):
    forecasts = []

    for i in range(len(data) - window_size - forecast_days + 1):
        try:
            window = data[i:i + window_size]
            actual_values = data['Price'][i + window_size:i + window_size + forecast_days]

            # Создаем и обучаем модель ARIMA
            model = ARIMA(window['Price'], order=(p, d, q))
            results = model.fit()

            # Делаем прогноз на 90 дней вперед
            forecast = results.get_forecast(steps=forecast_days)
            forecast_mean = forecast.predicted_mean.values

            # Рассчитываем относительное отклонение в процентах (Percentage Error)
            percentage_error = ((forecast_mean - actual_values) / actual_values) * 100

            # Добавляем результаты в список
            result = {
                'p': p,
                'd': d,
                'q': q,
                'begin_date': window['Date'].iloc[0],
                'end_date': window['Date'].iloc[-1] + pd.Timedelta(days=forecast_days - 1),
                'window_size': window_size,
                'forecast_days': forecast_days,
                'forecast_precision': np.mean(np.abs(percentage_error))
            }
            forecasts.append(result)
        except:
            continue

    return forecasts

def load_and_transform_data():
    #load data
    data = pd.read_csv("..\..\..\Data\Day\S&P 500 Historical Data00-20.csv")
    #transform data
    data = data.replace(",", "", regex=True)
    data.Date = pd.to_datetime(data.Date)
    data = data.sort_values(by=["Date"])
    data.set_index('Date')
    data.Price = data.Price.astype(float)
    data.High = data.High.astype(float)
    data.Low = data.Low.astype(float)
    data.Open = data.Open.astype(float)
    data["Vol."] = data["Vol."].astype(float)
    # раз
    data.index = data.index[::-1]
    #
    special_data = data[(data['Date'] > '2010-01-01') & (data['Date'] < '2014-01-01')]
    special_data.index = data.index[:len(special_data)]
    #
    special_data = special_data.drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'])
    return special_data

def save_result(results_df, p_begin, d_begin, q_begin, iteration_num):
    results_df.to_csv(f'output{p_begin}_{d_begin}_{q_begin}_{iteration_num}.csv', encoding='utf8')

def calculate_ARIMA(p_begin, d_begin, q_begin, iteration_num):
    # Указываем наши промежутки
    p_values = range(p_begin, 11)
    d_values = range(d_begin, 11)
    q_values = range(q_begin, 11)
    # Получаем данные
    special_data = load_and_transform_data()
    # Создаем пустой DataFrame для сохранения результатов
    columns = ['p', 'd', 'q', 'begin_date', 'end_date', 'window_size', 'forecast_days', 'forecast_precision']
    results_df = pd.DataFrame(columns = columns)
    counter = 0
    print(special_data.head(1))
    # Проходим по всем комбинациям p, d, q и строим прогнозы
    for p in p_values:
        for d in d_values:
            for q in q_values:
                if counter == iteration_num:
                    print(results_df.head(2))
                    save_result(results_df, p_begin, d_begin, q_begin, iteration_num)
                    exit(0)
                counter+=1
                print(f'Calculating:({p}, {d}, {q})')
                # Строим прогноз для текущих значений p, d, q
                forecasts = build_arima_model_and_forecast(special_data, p, d, q)
                # Добавляем результаты в DataFrame
                results_df = pd.concat([results_df, pd.DataFrame(forecasts)], ignore_index = True)#results_df.append(forecasts, ignore_index=True)


def main():
    warnings.filterwarnings('ignore')
    # Проверим, что передано четыре аргумента
    if len(sys.argv) != 5:
        print("Использование: python myscript.py <p> <d> <q> <iter_num>")
        return

    try:
        # Преобразуем аргументы командной строки в целые числа
        num1 = int(sys.argv[1])
        num2 = int(sys.argv[2])
        num3 = int(sys.argv[3])
        num4 = int(sys.argv[4])

        calculate_ARIMA(num1, num2, num3, num4)
    except ValueError as ve:
        print(f"Ошибка: {ve}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
