import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# Function to remove the dollar signs from a NASDAQ stock data csv that was with read_csv
def remove_dollar_signs(data):
    data["Close/Last"] = data["Close/Last"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["Open"] = data["Open"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["High"] = data["High"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["Low"] = data["Low"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    return data

#Plot function for the EDA
def plot_closing_price(data, stockname):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Close Prices')
    plt.plot(data['Close/Last'])
    title = stockname + " closing price"
    plt.title(title)
    save_path = "../FIGURES/" + stockname + "_Close_Price_Plot"
    plt.savefig(save_path)
    plt.close()

# Load the dataset and clean it
def load_dataset(stockname):
    filepath = '../DATA/' + stockname + '_Data.csv'
    data = remove_dollar_signs(pd.read_csv(filepath, sep=',', index_col='Date', parse_dates=['Date']))
    return data[::-1]

def plot_probability_distribution(data, stockname):
    df_close = data['Close/Last']
    df_close.plot(kind='kde')

# Loading in dataset
AAPL = load_dataset("AAPL")
DAL = load_dataset("DAL")
F = load_dataset("F")
FUN = load_dataset("FUN")
GME = load_dataset("GME")
TSLA = load_dataset("TSLA")

#plot close price
plot_closing_price(AAPL, "AAPL")
plot_closing_price(DAL, "DAL")
plot_closing_price(F, "F")
plot_closing_price(FUN, "FUN")
plot_closing_price(GME, "GME")
plot_closing_price(TSLA, "TSLA")

#plot_probability_distribution(AAPL, "AAPL")

#Test for staionarity
def test_stationarity(timeseries, ticker):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation of ' + ticker)
    filepath = "../FIGURES/"+ticker+"_stationarity"
    plt.savefig(filepath)
    plt.show(block=False)
    print("Results of dickey fuller test for", ticker)
    plt.close()
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

test_stationarity(AAPL['Close/Last'], "AAPL")
test_stationarity(DAL['Close/Last'], "DAL")
test_stationarity(F['Close/Last'],"F" )
test_stationarity(FUN['Close/Last'], "FUN")
test_stationarity(GME['Close/Last'], "GME")
test_stationarity(TSLA['Close/Last'], "TSLA")


#eliminate trend if not stationary
from matplotlib import rcParams
def eliminate_trends(data,stockname): 
    rcParams['figure.figsize'] = 10, 6
    df_log = np.log(data['Close/Last'])
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()
    plt.title('Moving Average')
    plt.plot(std_dev, color ="black", label = "Standard Deviation")
    plt.plot(moving_avg, color="red", label = "Mean")
    plt.legend()
    filepath = "../FIGURES/" + stockname + "_Eliminate_Trends_Plot"
    plt.savefig(filepath)
    plt.close()
    return df_log

#AAPL = eliminate_trends(AAPL, "AAPL")
#DAL = eliminate_trends(DAL, "DAL")
#F = eliminate_trends(F, "F")
#FUN = eliminate_trends(FUN, "FUN")
#GME = eliminate_trends(GME, "GME")
#TSLA = eliminate_trends(TSLA, "TSLA")

AAPL = AAPL['Close/Last']
DAL = DAL['Close/Last']
F = F['Close/Last']
FUN = FUN['Close/Last']
GME = GME['Close/Last']
TSLA = TSLA['Close/Last']

#looks specifically to split on the date 
def split_data(df_log, ticker):
    train_data, test_data = df_log[:len(df_log)-260], df_log[len(df_log)-260:]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    filepath = "../FIGURES/" + ticker + "_test_train_split"
    plt.savefig(filepath)
    return train_data, test_data

AAPL_train, AAPL_test = split_data(AAPL, "AAPL")
DAL_train, DAL_test = split_data(DAL, "DAL")
F_train, F_test = split_data(F, "F")
FUN_train, FUN_test = split_data(FUN, "FUN")
GME_train, GME_test = split_data(GME, "GME")
TSLA_train, TSLA_test = split_data(TSLA, "TSLA")

def find_model_params(train_data):
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
        test='adf',       # use adftest to find optimal 'd'
        max_p=5, max_q=5, # maximum p and q
        m=1,              # frequency of series
        d=None,           # let model determine 'd'
        seasonal=False,   # No Seasonality
        start_P=0, 
        D=0, 
        trace=True,
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True)
    return model_autoARIMA

AAPL_model = find_model_params(AAPL_train) #1,1,0
print(AAPL_model.summary())
DAL_model = find_model_params(DAL_train) #3,1,0
print(DAL_model.summary())
F_model = find_model_params(F_train) #0,1,0
print(F_model.summary())
FUN_model = find_model_params(FUN_train) #1,1,0
print(FUN_model.summary())
GME_model = find_model_params(GME_train) #0,1,0
print(GME_model.summary())
TSLA_model = find_model_params(TSLA_train) #0,1,0
print(TSLA_model.summary())

def train_and_fit_model(data, order):
    model = ARIMA(data, order=order)  
    fitted = model.fit()  
    print(fitted.summary())
    return fitted

print("BREAK\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

AAPL_fitted = train_and_fit_model(AAPL_train, (1,1,0))
DAL_fitted = train_and_fit_model(DAL_train, (3,1,0))
F_fitted = train_and_fit_model(F_train, (0,1,0))
FUN_fitted = train_and_fit_model(FUN_train, (1,1,0))
GME_fitted = train_and_fit_model(GME_train, (0,1,0))
TSLA_fitted = train_and_fit_model(TSLA_train, (0,1,0))


def forecast_and_analyze(train_data, test_data, fitted, ticker):
    # Forecast
    fc = fitted.forecast(len(test_data), alpha=0.05)  # 95% conf

    print(type(test_data))
    print("test_data:", test_data)
    print("fc:", fc)

    # Make as pandas series
    #fc_series = pd.Series(fc, index=test_data.index)
    #combined_series = pd.Series(fc.values, index=test_data.index + pd.to_timedelta(fc.index, unit='D'))
    # Forecasting the next 'n' steps
    n_steps = 260
    forecast_result = fitted.get_forecast(steps=n_steps)

    # Extract the forecast and the confidence interval
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

    combined_series = pd.Series(fc.values, index=test_data.index)

    # lower_series = pd.Series(conf[:, 0], index=test_data.index)
    # upper_series = pd.Series(conf[:, 1], index=test_data.index)
    # Plot
    print("fc_series: ", combined_series)
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(train_data, label='training data')
    plt.plot(test_data, color = 'blue', label='Actual Stock Price')
    plt.plot(combined_series, color = 'orange',label='Predicted Stock Price')
    #plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
    plt.fill_between(combined_series.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='k', alpha=0.1, label='95% Confidence Interval')
    plt.title(ticker+ ' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(ticker + ' Stock Price')
    plt.legend(loc='upper left', fontsize=8)
    filepath = "../FIGURES/"+ticker+"_prediction"
    plt.savefig(filepath)

    # report performance
    print(ticker + " performance:")
    mse = mean_squared_error(test_data, fc)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test_data, fc)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_data, fc))
    print('RMSE: '+str(rmse))
    # mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
    # print('MAPE: '+str(mape))

forecast_and_analyze(AAPL_train, AAPL_test, AAPL_fitted, "AAPL")
forecast_and_analyze(DAL_train, DAL_test, DAL_fitted, "DAL")
forecast_and_analyze(F_train, F_test, F_fitted, "F")
forecast_and_analyze(FUN_train, FUN_test, FUN_fitted, "FUN")
forecast_and_analyze(GME_train, GME_test, GME_fitted, "GME")
forecast_and_analyze(TSLA_train, TSLA_test, TSLA_fitted, "TSLA")


