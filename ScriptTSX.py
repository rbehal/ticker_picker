import sys
import os

try:
    os.chdir(os.path.join(os.getcwd(), "ticker_picker"))
    print(os.getcwd())
except:
    pass

import requests
import json
import pandas as pd
import numpy as np
import ta
import time
from pandasgui import show
import tkinter
from tkinter import *
from tkinter import messagebox

# Daily Timeframe Data Grabber


def daily_df(ticker):

    API_URL = "https://www.alphavantage.co/query"

    data = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "TSX:" + ticker,
        "outputsize": "compact",
        "datatype": "json",
        "apikey": "PVGTC9P6IUS7XLQ9",
    }

    response = requests.get(API_URL, params=data)

    if response.status_code == 200:
        stock_data = response.json()
        error_check = list(stock_data.keys())
        if error_check[0] == "Error Message":
            raise SyntaxError("Ticker does not exist.")
        else:
            meta_data = stock_data["Meta Data"]
            daily_data = stock_data["Time Series (Daily)"]
            daily_dataframe_rev = pd.DataFrame.from_dict(daily_data)
            daily_dataframe = daily_dataframe_rev.transpose()
            daily_dataframe.columns = ["open", "high", "low", "close", "volume"]
            df_entries2float(daily_dataframe)
            return daily_dataframe
    else:
        return response.status_code


# Converting dataframe entries to floats


def df_entries2float(df):
    for i in df.columns:
        column2convert = df[i]
        column_size = column2convert.size
        for k in range(0, column_size):
            column2convert[k] = float(column2convert[k])
    return df


# Reversing dataframe


def rev(df):
    return df.sort_index(ascending=True, axis=0)


# Data gatherer
# Rahul's Computer Ticker String Data: "C:/Users/Rahul Behal/OneDrive/Documents/Programming/yHacks/ticker_picker/Stock Data/tsx_data/tsx_stocks.csv"
# Rahul's Computer Ticker Stock Dataframes: "C:/Users/Rahul Behal/OneDrive/Documents/Programming/yHacks/ticker_picker/Stock Data/tsx_data/"
def get_data(filepath_tickers, filepath_data):
    """
        filepath_tickers: Filepath of a CSV of ticker names.
        filepath_data: Where you want the OHLCV data csv files to be stored.
        If program stops, reset count value to what it says and restart the function.
    """
    raw_data = pd.read_csv(filepath_tickers)
    count = 99  # Value to be reset

    stocks_in_tsx_raw = raw_data.columns[count:]
    stocks_in_tsx_proper = []

    for i in stocks_in_tsx_raw:
        fixed = i.strip()
        stocks_in_tsx_proper.append(fixed)

    for i in stocks_in_tsx_proper:
        try:
            time.sleep(12)
            stock_data = daily_df(i)
            stock_data.to_csv(filepath_data + i + ".csv")
            count += 1
        except KeyError:
            print("Retry count {}, ticker {}.".format(count, i))
            count += 1
            break
        except SyntaxError:
            print("{} invalid ticker, start at {}".format(i, count + 1))
            break


# Running TA with proper formatting


def tech_indicator(indicator, daily_df):
    indicator_df = pd.DataFrame(indicator)
    indicator_df = indicator_df.sort_index(ascending=False, axis=0)
    resulting_df = daily_df.join(indicator_df)
    return resulting_df


# All technical indicators in use


def run_indicators(df):
    high = rev(df["high"])
    low = rev(df["low"])
    close = rev(df["close"])
    volume = rev(df["volume"])

    df = tech_indicator(ta.momentum.wr(high, low, close), df)
    df = tech_indicator(ta.momentum.money_flow_index(high, low, close, volume), df)
    df = tech_indicator(ta.momentum.stoch_signal(high, low, close), df)
    df = tech_indicator(ta.momentum.tsi(close), df)
    df = tech_indicator(ta.trend.macd(close), df)
    df = tech_indicator(ta.trend.trix(close), df)
    df = tech_indicator(ta.trend.aroon_up(close), df)
    df = tech_indicator(ta.trend.aroon_down(close), df)
    df = tech_indicator(
        ta.volatility.keltner_channel_hband_indicator(high, low, close), df
    )
    df = tech_indicator(
        ta.volatility.keltner_channel_lband_indicator(high, low, close), df
    )
    df = tech_indicator((ta.volatility.bollinger_hband_indicator(close)), df)
    df = tech_indicator((ta.volatility.bollinger_lband_indicator(close)), df)
    df = tech_indicator(ta.volume.chaikin_money_flow(high, low, close, volume), df)

    df["bbi"] = df["bbihband"].subtract(df["bbilband"])
    df["kci"] = df["kci_hband"].subtract(df["kci_lband"])
    del df["bbihband"]
    del df["bbilband"]
    del df["kci_hband"]
    del df["kci_lband"]
    df = tech_indicator(ta.momentum.rsi(close), df)

    return df


# Mapping values


def map_values(df):
    length = len(df)

    df = df.copy()

    #############
    ####TREND####
    #############

    # Mapping MACD values
    macd = df.columns.get_loc("MACD_12_26")
    for i in range(length):
        if 0 < df.iloc[i, macd]:
            df.iloc[i, macd] = 1
        elif df.iloc[i, macd] < 0:
            df.iloc[i, macd] = -1
        else:
            df.iloc[i, macd] = 0

    # Mapping TRIX values
    trix = df.columns.get_loc("trix_15")
    for i in range(length):
        if 0 < df.iloc[i, trix]:
            df.iloc[i, trix] = 1
        elif df.iloc[i, trix] < 0:
            df.iloc[i, trix] = -1
        else:
            df.iloc[i, trix] = 0

    # Mapping Aroon values
    values = []
    aroon_up = df.columns.get_loc("aroon_up25")
    aroon_down = df.columns.get_loc("aroon_down25")
    for i in range(length):
        if df.iloc[i, aroon_up] >= 70 and df.iloc[i, aroon_down] <= 30:
            values.append(2.0)
        elif df.iloc[i, aroon_down] >= 70 and df.iloc[i, aroon_up] <= 30:
            values.append(-2.0)
        elif df.iloc[i, aroon_up] > df.iloc[i, aroon_down]:
            values.append(1.0)
        elif df.iloc[i, aroon_down] > df.iloc[i, aroon_up]:
            values.append(-1.0)
        else:
            values.append(0.0)
    df["aroon"] = values

    # Mapping Chaikin values
    cmf = df.columns.get_loc("cmf")
    for i in range(length):
        if 0.05 <= df.iloc[i, cmf] < 0.1:
            df.iloc[i, cmf] = 1
        elif -0.1 < df.iloc[i, cmf] <= -0.05:
            df.iloc[i, cmf] = -1
        elif df.iloc[i, cmf] >= 0.1:
            df.iloc[i, cmf] = 2
        elif df.iloc[i, cmf] <= -0.1:
            df.iloc[i, cmf] = -2
        else:
            df.iloc[i, cmf] = 0

    #############
    # OSCILLATORS#
    #############

    # Mapping MFI values
    mfi = df.columns.get_loc("mfi_14")
    for i in range(length):
        if 0 <= df.iloc[i, mfi] <= 20:
            df.iloc[i, mfi] = 2
        elif 80 <= df.iloc[i, mfi] <= 100:
            df.iloc[i, mfi] = -2
        elif 20 <= df.iloc[i, mfi] < 30:
            df.iloc[i, mfi] = 1
        elif 70 < df.iloc[i, mfi] <= 80:
            df.iloc[i, mfi] = -1
        else:
            df.iloc[i, mfi] = 0

    # Mapping rsi values
    rsi = df.columns.get_loc("rsi")
    for i in range(length):
        if 70 <= df.iloc[i, rsi] < 100:
            df.iloc[i, rsi] = -2
        elif 0 <= df.iloc[i, rsi] <= 30:
            df.iloc[i, rsi] = 2
        elif 30 < df.iloc[i, rsi] <= 40:
            df.iloc[i, rsi] = 1
        elif 60 <= df.iloc[i, rsi] < 70:
            df.iloc[i, rsi] = -1
        else:
            df.iloc[i, rsi] = 0

    # Mapping Stoch_d values
    stoch = df.columns.get_loc("stoch_d")
    for i in range(length):
        if 80 <= df.iloc[i, stoch] <= 100:
            df.iloc[i, stoch] = -2
        elif 0 <= df.iloc[i, stoch] <= 20:
            df.iloc[i, stoch] = 2
        elif 20 < df.iloc[i, stoch] <= 30:
            df.iloc[i, stoch] = 1
        elif 70 <= df.iloc[i, stoch] < 80:
            df.iloc[i, stoch] = -1
        else:
            df.iloc[i, stoch] = 0

    # Mapping TSI values
    tsi = df.columns.get_loc("tsi")
    for i in range(length):
        if df.iloc[i, tsi] < -25:
            df.iloc[i, tsi] = -2
        elif df.iloc[i, tsi] <= -10:
            df.iloc[i, tsi] = -1
        elif df.iloc[i, tsi] <= 10:
            df.iloc[i, tsi] = 0
        elif df.iloc[i, tsi] < 25:
            df.iloc[i, tsi] = 1
        else:
            df.iloc[i, tsi] = 2

    # Mapping wr values
    wr = df.columns.get_loc("wr")
    for i in range(length):
        if -100 <= df.iloc[i, wr] <= -80:
            df.iloc[i, wr] = 2
        elif -20 <= df.iloc[i, wr] <= 0:
            df.iloc[i, wr] = -2
        elif -30 <= df.iloc[i, wr] < -20:
            df.iloc[i, wr] = -1
        elif -80 < df.iloc[i, wr] <= -70:
            df.iloc[i, wr] = 1
        else:
            df.iloc[i, wr] = 0

    del df["aroon_up25"]
    del df["aroon_down25"]

    df["Overall Rating"] = (
        df["wr"]
        + df["mfi_14"]
        + df["stoch_d"]
        + df["tsi"]
        + df["MACD_12_26"]
        + df["trix_15"]
        + df["cmf"]
        + df["bbi"]
        + df["kci"]
        + df["rsi"]
        + df["aroon"]
    )

    return df


# Returning mapped indicator data less OHLCV
def indicator_data(ticker):
    df = daily_df(ticker)
    df = run_indicators(df)
    df = map_values(df)
    df = df[:-45]
    df = df.rename(
        columns={
            "wr": "William's PR",
            "mfi_14": "Money Flow Index",
            "stoch_d": "Stochastic Oscilattor",
            "tsi": "True Strength Index",
            "MACD_12_26": "MACD",
            "trix_15": "Trix",
            "cmf": "Chaikin Money Flow",
            "bbi": "Bollinger Bands",
            "kci": "Keltner Channel",
            "rsi": "Relative Strength Index",
            "aroon": "Aroon",
        }
    )
    del df["open"]
    del df["close"]
    del df["volume"]
    del df["low"]
    del df["high"]
    df = df.astype(float)
    df = df.astype(str)
    for i in range(len(df)):
        for k in range(0, 11):
            if df.iloc[i][k] == str(-2.0):
                df.iloc[i][k] = "Strong Sell"
            elif df.iloc[i][k] == str(-1.0):
                df.iloc[i][k] = "Sell"
            elif df.iloc[i][k] == str(0.0):
                df.iloc[i][k] = "Neutral"
            elif df.iloc[i][k] == str(1.0):
                df.iloc[i][k] = "Buy"
            elif df.iloc[i][k] == str(2.0):
                df.iloc[i][k] = "Strong Buy"
    return df  # Returns indicator signals and consensus


def stock_predict(ticker):
    df = daily_df(ticker)
    df = run_indicators(df)
    df = map_values(df)
    df = df[:-45]
    return df


def stock_predict_HC(df):
    df = run_indicators(df)
    df = map_values(df)
    df = df[:-45]
    return df


def ao3(df):
    ao3 = []

    for i in range(len(df.columns)):

        sum = 0

        for j in range(3):
            sum += df.iloc[j][i]

        average = sum / 3
        ao3.append(average)

    df = pd.DataFrame(np.array([ao3]), columns=df.columns).append(
        df, ignore_index=False
    )
    df = df.rename(index={0: "Average of 3"})
    df = df.sort_values(by=[df.index[0]], axis=1, ascending=False)
    return df


# Creating dataframe of TSX stock data ratings

# Enter filepaths here; first for list of tickers, second for local storage of files
filepath_tickers = "C:/Users/Rahul Behal/OneDrive/Documents/Programming/yHacks/ticker_picker/Stock Data/tsx_data/tsx_stocks.csv"
filepath_data = "C:/Users/Rahul Behal/OneDrive/Documents/Programming/yHacks/ticker_picker/Stock Data/tsx_data/"
raw_data = pd.read_csv(filepath_tickers)

stocks_in_tsx_raw = raw_data.columns[:]
stocks_in_tsx_proper = []

# Removing any whitespace from ticker headings
for i in stocks_in_tsx_raw:
    fixed = i.strip()
    stocks_in_tsx_proper.append(fixed)

# Starting congregation of overall ratings of tickers
ticker = stocks_in_tsx_proper[0]
# All raw stock data
raw_data = pd.read_csv(filepath_data + ticker + ".csv", index_col=0)
# Predicting the stock data
predicted_data_raw = stock_predict_HC(raw_data)
# Splicing overall rating for the particular stock
predicted_data_preprocessed = pd.DataFrame(predicted_data_raw["Overall Rating"])
# Creating final datatable with ticker name and all
predicted_data_abs_final = predicted_data_preprocessed.rename(
    columns={"Overall Rating": ticker}
)

# Doing the same thing in a loop, but joining to the newly created conjugate df
for i in range(1, len(stocks_in_tsx_proper)):
    try:
        ticker = stocks_in_tsx_proper[i]
        raw_data = pd.read_csv(filepath_data + ticker + ".csv", index_col=0)
        predicted_data_raw = stock_predict_HC(raw_data)
        predicted_data_preprocessed = pd.DataFrame(predicted_data_raw["Overall Rating"])
        predicted_data_final = predicted_data_preprocessed.rename(
            columns={"Overall Rating": ticker}
        )
        predicted_data_abs_final = predicted_data_abs_final.join(predicted_data_final)
    except:
        continue

# Sorting final data collection of aggregate ratings highest to lowest
predicted_data_abs_final = predicted_data_abs_final.sort_values(
    by=[predicted_data_abs_final.index[0]], axis=1, ascending=False
)

# Dropping all n/a data
predicted_data_abs_final = predicted_data_abs_final.dropna(axis=1)

# Final local database of stored indicator values
data_averaged = ao3(predicted_data_abs_final.copy())

# Retrieve the top 20 buys/sells based on the 10 day average indicator value.
def top20(df, position="buy"):
    if position == "buy":
        top20_buy = pd.DataFrame(df.iloc[0, :20])
        top20_buy = top20_buy.reset_index()
        top20_buy = top20_buy.rename(
            columns={"index": "Ticker", "Average of 3": "Score"}
        )
        top20_buy.index += 1
        return top20_buy
    if position == "sell":
        top20_sell = pd.DataFrame(df.iloc[0, -20:])
        top20_sell = top20_sell.rename(
            columns={"index": "Ticker", "Average of 3": "Score"}
        )
        top20_sell = top20_sell.sort_values(by=["Score"], axis=0, ascending=True)
        top20_sell = top20_sell.reset_index()
        top20_sell.index += 1
        top20_sell = top20_sell.rename(columns={"index": "Ticker"})
        return top20_sell


# Retrieves and displays the raw data for the specified ticker.
def get_ticker(ticker: str):
    try:
        df = daily_df(ticker)
        df = df.rename(
            columns={
                "open": "OPEN",
                "high": "HIGH",
                "low": "LOW",
                "close": "CLOSE",
                "volume": "VOLUME",
            }
        )
        params = {str(locals()["ticker"]): df}
        show(**params)
    except:
        messagebox.showerror("Error!", "Invalid ticker entered!")


# Retrieves and displays the indicator values for the specified ticker.
def get_indicators(ticker: str):

    df = indicator_data(ticker)
    params = {str(locals()["ticker"]): df}
    show(**params)


# Display the top 20 buys for the present day based on a 10 day average.
def get_buy20():
    df = top20(data_averaged)
    params = {"Top 20 Buys": df}
    show(**params)


# Display the top 20 sells for the present day based on a 10 day average.
def get_sell20():
    df = top20(data_averaged, "sell")
    params = {"Top 20 Sells": df}
    show(**params)


# GUI implementation. All elements are created in order of their display from top/left to bottom/right.
top = tkinter.Tk()
top.title("Ticker Picker")
top.width = "15c"

topframe = Frame(top)
topframe.pack()
label = tkinter.Label(
    topframe,
    text="Enter a ticker to retrieve that stock's data or it's buy/sell indicator status.",
)
label.pack()

frame = Frame(top)
frame.pack()
val = tkinter.Entry(frame)
val.pack(side=RIGHT)

bottomframe = Frame(top)
bottomframe.pack()
best20 = tkinter.Button(bottomframe, text="Top 20 Buys", command=get_buy20)
worst20 = tkinter.Button(bottomframe, text="Top 20 Sells", command=get_sell20)
best20.pack(side=LEFT)
worst20.pack(side=LEFT)
indicatorbutton = tkinter.Button(
    bottomframe,
    text="Retrieve indicator summary",
    command=lambda: get_indicators(val.get().upper()),
)
databutton = tkinter.Button(
    bottomframe, text="Retrieve raw data", command=lambda: get_ticker(val.get().upper())
)
databutton.pack(side=RIGHT)
indicatorbutton.pack(side=RIGHT)

legend = Frame(top)
legend.pack(side=BOTTOM)
label = tkinter.Label(
    legend,
    text="Strong Buy (8 to 16)\nBuy (1 to 8)\nNeutral (-1 to 1)\nSell (-8 to -1)\nStrong Sell (-16 to -8)",
    borderwidth=2,
    relief="solid",
)
label.pack(side=LEFT)

top.mainloop()
