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

#split_fraction = 0.715
train_split = 2262 # int(split_fraction * int(AAPL.shape[0]))
step = 30

past = 2258
future = 255
learning_rate = 0.001
batch_size = 805
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

AAPL = normalize(AAPL, train_split)
DAL = normalize(DAL, train_split)
F = normalize(F, train_split)
FUN = normalize(FUN, train_split)
GME = normalize(GME, train_split)
TSLA = normalize(TSLA, train_split)


AAPL = pd.DataFrame(AAPL)
print(AAPL.head())
#features.index = df[date_time_key]

AAPL_train = AAPL.iloc[0 : train_split]
AAPL_test = AAPL.iloc[train_split:]

start = 0
end = 2262

# This will select the 1st, 2nd, 3rd, and 4th columns (0-based index)

x_train = AAPL_train[['Close/Last','Volume']]
y_train = AAPL.iloc[start:end][['Close/Last']]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(AAPL_test) + train_split

label_start = train_split

x_val = AAPL_test.iloc[:x_end][['Close/Last','Volume']].values
y_val = AAPL.iloc[label_start:][['Close/Last']]
#print("start of x")
#print(x_val)
#print("start of y")
#rint(y_val)
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)


inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
)