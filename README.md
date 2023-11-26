# Project3

## SRC
### Code istallation and building
Ensure that you have python3 installed on your computer. While not required, it is also suggested that you run the following installation commands within a python virtual environment. Information about setting up a virtual environment can be found [here](https://docs.python.org/3/library/venv.html).

Install the following python libraries in order to run the code
- ```pip install pmdarmima```
- ```pip install matplotlib```
- ```pip install pylab-sdk```
- ```pip install pandas```
- ```pip install statsmodels```

### Code Usage
To run our model use the command ```python3 model.py``` from within the SRC folder. You must also create a FIGURES folder at the same level as the SRC folder beforehand so the charts save properly. Your data must be stored in the same format as the provided data and have the filename TICKER_Data.csv. Note in the code our test and train is split manually rather than by a certain portion of the data. If a different split is desired, the code in `split_data` must be modified accordingly. 

## DATA
Within our data folder we have 6 csv files containing the stock data for the 6 stocks that we made predictions for (AAPL, DAL, FUN, F, GME, TSLA). This data is taken from the NASDAQ website [1]. The dates for each stock range from 11/12/2013 to 11/10/2023. Descriptions of the specific attributes for each data entry are in the data dictionary below. NASDAQ only allows access to the last 10 years of data, so data downloaded from the site after 11/10/2023 will not be able to go back as far as the data stored in this repository. 

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
| TICKER_Close_Price_Plot.png | Plot of the stock's daily closing price from 11/12/2013 to 11/10/2023|
| TICKER_prediction.png | Plot of the real closing stock price for the entire data, along with our models prediction and confidence intervals for the testing data |
| TICKER_Predicted_Prices.png | Plot of the predicted stock prices for each month from 11/01/2022 to 11/01/2023|
| TICKER_stationarity.png | Plot exploring the initial stationarity of the data before modifying the data to make it stationary | 
| TICKER_test_train_split.png | Plot of the actual closing stock price from 11/12/2013 to 11/10/2023, with the colors indicating where the training and testing data starts and stops  | 
| TICKER_Autocorrelation.png | Plot that checks if there's auto-correlation in the stock data, a linear trend means the ARIMA model is appropriate|



## REFERENCES
[1] “Historical data,” Nasdaq, https://www.nasdaq.com/market-activity/quotes/historical (accessed Nov. 3, 2023). \
[2] H. Dhaduk, “Stock market forecasting using time series analysis with Arima Model,” Analytics Vidhya, https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/ (accessed Nov. 8, 2023).\
[3] P. Serafeim Loukas, “Time-series forecasting: Predicting stock prices using an Arima model,” Medium, https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70 (accessed Nov. 26, 2023).\
[4]</a> [Milestone 1 Document](https://docs.google.com/document/d/1y7nvNOVDFKOomJsZwtJEGYkxmL8Pg9dptXSihSPh5Zg/edit) \
[5]</a> [Milestone 2 Document](https://docs.google.com/document/d/1u58Wji9ejL_7v4YxKf8v2M9AJXf7ctSlRfmaztsKcjc/edit)

