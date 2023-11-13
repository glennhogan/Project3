import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
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
    return remove_dollar_signs(pd.read_csv(filepath, sep=',', index_col='Date', parse_dates=['Date']))

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
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.savefig('../FIGURES/test1')
    plt.show(block=False)
    print("Results of dickey fuller test")
    plt.close()
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

test_stationarity(AAPL['Close/Last'])

#eliminate trend if not stationary
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(AAPL['Close/Last'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.savefig('../FIGURES/test2')
plt.close()