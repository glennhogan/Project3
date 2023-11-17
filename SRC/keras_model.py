import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


#data cleaning
def remove_dollar_signs(data):
    data["Close/Last"] = data["Close/Last"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["Open"] = data["Open"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["High"] = data["High"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    data["Low"] = data["Low"].replace({'\$': ''}, regex=True).apply(pd.to_numeric, errors='ignore')
    return data

def load_dataset(stockname):
    filepath = '../DATA/' + stockname + '_Data.csv'
    data = remove_dollar_signs(pd.read_csv(filepath, sep=',', index_col='Date', parse_dates=['Date']))
    return data[::-1]

AAPL = load_dataset("AAPL")
DAL = load_dataset("DAL")
F = load_dataset("F")
FUN = load_dataset("FUN")
GME = load_dataset("GME")
TSLA = load_dataset("TSLA")

split_fraction = 0.715
train_split = int(split_fraction * int(AAPL.shape[0]))
step = 30

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

