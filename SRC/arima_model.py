import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70
# pip install statsmodels

def remove_dollar_signs(data):
    data["Close/Last"] = data["Close/Last"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["Open"] = data["Open"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["High"] = data["High"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["Low"] = data["Low"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    return data

def load_dataset(stockname):
    #filepath = '../DATA/' + stockname + '_Data.csv'
    filepath = 'DATA/' + stockname + '_Data.csv'
    data = remove_dollar_signs(pd.read_csv(filepath, sep=',', index_col='Date', parse_dates=['Date']))
    return data[::-1]

def check_cross_correlation(df, stockname):
    plt.figure()
    lag_plot(df['Open'], lag=3)
    plt.title(f'{stockname} Stock - Autocorrelation plot with lag = 3')
    plt.savefig("../FIGURES/" + stockname + "_Autocorrelation")
    plt.close()

def plot_price_evolution(df, stockname):
    january_1st_dates = pd.date_range(start='2013-01-01', end='2023-01-01', freq='YS')
    plt.plot(df.index, df["Close/Last"])
    plt.xticks(january_1st_dates, [date.strftime('%Y') for date in january_1st_dates])
    plt.title(f"{stockname} stock price over time")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.savefig("../FIGURES/" + stockname + "_Price_Evolution")
    plt.close()

def build_arima_model(df, stockname):
    start_date = '2022-11-01'
    end_date = '2023-11-10'
    #end_date = '2022-12-01'
    test = df.loc[start_date:end_date]
    train = df.loc[~((df.index >= start_date) & (df.index <= end_date))]
    training_data = train['Close/Last'].values
    test_data = test['Close/Last'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    RMSE_error = np.sqrt(MSE_error)
    print('Testing Mean Squared Error is {}'.format(MSE_error))
    print('Testing Root Mean Squared Error is {}'.format(RMSE_error))
    test_set_range = test.index
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title(f'{stockname} Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='MS')
    plt.xticks(monthly_ticks, [date.strftime('%m-%d-%Y') for date in monthly_ticks])
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.savefig("../FIGURES/" + stockname + "_Predicted_Prices")
    plt.close()
    

def get_predictions(stockname):
    df = load_dataset(stockname)
    check_cross_correlation(df, stockname)
    plot_price_evolution(df, stockname)
    build_arima_model(df, stockname)

get_predictions("AAPL")
get_predictions("DAL")
get_predictions("F")
get_predictions("FUN")
get_predictions("GME")
get_predictions("TSLA")
