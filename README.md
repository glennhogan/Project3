# Project3

## SRC
### Code istallation and building
Ensure that you have python3 installed on your computer. While not required, it is also suggested that you run the following installation commands within a python virtual environment. Information about setting up a virtual environment can be found [here](https://docs.python.org/3/library/venv.html).

- ```pip install pmdarmima```
- ```pip install matplotlib```
- ```pip install pylab-sdk```

### Code Usage
To run our model use the command ```python3 SRC/model.py```.\ 

## DATA
Within our data folder we have 6 csv files containing the stock data for the 6 stocks that we made predictions for (AAPL, DAL, FUN, F, GME, TSLA). 

### Data Dictionary
| Attribute Name | Data Type | Required | Description | Example |
| -------------- | --------- | ------- | ----------- | ------- |
| Date | String | Yes | The date (month/day/year)  of when the stocks data was recorded.| 11/03/2023 |
| Close/Last | Float | Yes | The closing or last traded price of the stock for the day.| 219.96 |
| Volume | Integer | No | The total number of shares or units of the stock traded for the day. | 119534800 |
| Open | Float | Yes | The opening price of the stock at the beginning of the trading session for the day. | 221.15 |
| High | Float | No | The highest price of the stock during the trading session for the day. | 226.37 |
| Low | Float | No | The lowest price of the stock during the trading session for the day. | 218.40 | 



## FIGURES
| Figure Name | Description | 
|-------------|-------------|
| TICKER_Close_Price_Plot.png | Plot of the stock's daily closing price|
| TICKER_Eliminate_Trends_Plot.png | tmp |
| TICKER_prediction.png | tmp |
| TICKER_stationarity.png | tmp | 
| TICKER_test_train_split.png | tmp |



## REFERENCES
[1] “Historical data,” Nasdaq, https://www.nasdaq.com/market-activity/quotes/historical (accessed Nov. 3, 2023). \
[2] H. Dhaduk, “Stock market forecasting using time series analysis with Arima Model,” Analytics Vidhya, https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/ (accessed Nov. 8, 2023).\
[3]</a> [Milestone 1 Document](https://docs.google.com/document/d/1y7nvNOVDFKOomJsZwtJEGYkxmL8Pg9dptXSihSPh5Zg/edit) \
[4]</a> [Milestone 2 Document](https://docs.google.com/document/d/1u58Wji9ejL_7v4YxKf8v2M9AJXf7ctSlRfmaztsKcjc/edit)

