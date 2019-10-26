import sys
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'ticker_picker'))
    print(os.getcwd())
except:
    pass

import requests
import json
import pandas as pd
import ta
from pandasgui import show
import tkinter
from tkinter import messagebox

# Daily Timeframe Data Grabber


def daily_df(ticker):

    API_URL = "https://www.alphavantage.co/query"

    data = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",
        "datatype": "json",
        "apikey": "PVGTC9P6IUS7XLQ9",
    }

    response = requests.get(API_URL, params=data)

    if response.status_code == 200:
        stock_data = response.json()
        daily_data = stock_data['Time Series (Daily)']
        daily_dataframe_rev = pd.DataFrame.from_dict(daily_data)
        daily_dataframe = daily_dataframe_rev.transpose()
        daily_dataframe.columns = ['open', 'high', 'low', 'close', 'volume']
        df_entries2float(daily_dataframe)
        return daily_dataframe

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

# Running TA with proper formatting


def tech_indicator(indicator, daily_df):
    indicator_df = pd.DataFrame(indicator)
    indicator_df = indicator_df.sort_index(ascending=False, axis=0)
    resulting_df = daily_df.join(indicator_df)
    return resulting_df

# All technical indicators in use


def run_indicators(df):
    high = rev(df['high'])
    low = rev(df['low'])
    close = rev(df['close'])
    volume = rev(df['volume'])

    df = tech_indicator(ta.momentum.wr(high, low, close), df)
    df = tech_indicator(ta.momentum.money_flow_index(
        high, low, close, volume), df)
    df = tech_indicator(ta.momentum.stoch_signal(high, low, close), df)
    df = tech_indicator(ta.momentum.tsi(close), df)
    df = tech_indicator(ta.trend.macd(close), df)
    df = tech_indicator(ta.trend.trix(close), df)
    df = tech_indicator(ta.trend.aroon_up(close), df)
    df = tech_indicator(ta.trend.aroon_down(close), df)
    df = tech_indicator(
        ta.volatility.keltner_channel_hband_indicator(high, low, close), df)
    df = tech_indicator(
        ta.volatility.keltner_channel_lband_indicator(high, low, close), df)
    df = tech_indicator((ta.volatility.bollinger_hband_indicator(close)), df)
    df = tech_indicator((ta.volatility.bollinger_lband_indicator(close)), df)
    df = tech_indicator(ta.volume.chaikin_money_flow(
        high, low, close, volume), df)

    df['bbi'] = df['bbihband'].subtract(df['bbilband'])
    df['kci'] = df['kci_hband'].subtract(df['kci_lband'])
    del df['bbihband']
    del df['bbilband']
    del df['kci_hband']
    del df['kci_lband']
    df = tech_indicator(ta.momentum.rsi(close), df)

    return df

# Mapping values


def map_values(df):
    length = len(df)

    df = df.copy()

    # Mapping wr values
    wr = df.columns.get_loc('wr')
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

    # Mapping MFI values
    mfi = df.columns.get_loc('mfi_14')
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

    # Mapping Stoch_d values
    stoch = df.columns.get_loc('stoch_d')
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
    tsi = df.columns.get_loc('tsi')
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

    # Mapping MACD values
    macd = df.columns.get_loc('MACD_12_26')
    for i in range(length):
        if 0 < df.iloc[i, macd]:
            df.iloc[i, macd] = 1
        elif df.iloc[i, macd] < 0:
            df.iloc[i, macd] = -1
        else:
            df.iloc[i, macd] = 0

    # Mapping TRIX values
    trix = df.columns.get_loc('trix_15')
    for i in range(length):
        if 0 < df.iloc[i, trix]:
            df.iloc[i, trix] = 1
        elif df.iloc[i, trix] < 0:
            df.iloc[i, trix] = -1
        else:
            df.iloc[i, trix] = 0

    # Mapping Aroon values
    values = []
    aroon_up = df.columns.get_loc('aroon_up25')
    aroon_down = df.columns.get_loc('aroon_down25')
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
    df['aroon'] = values

    # Mapping Chaikin values
    cmf = df.columns.get_loc('cmf')
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

    # Mapping rsi values
    rsi = df.columns.get_loc('rsi')
    for i in range(length):
        if 70 <= df.iloc[i, rsi] < 100:
            df.iloc[i, rsi] = -2
        elif 0 <= df.iloc[i, rsi] <= 30:
            df.iloc[i, cmf] = 2
        elif 30 < df.iloc[i, rsi] >= 40:
            df.iloc[i, rsi] = 1
        elif 60 <= df.iloc[i, rsi] < 70:
            df.iloc[i, rsi] = -1
        else:
            df.iloc[i, rsi] = 0

    del df['aroon_up25']
    del df['aroon_down25']

    return df


def run_sample():
    df = daily_df('AAPL')
    df2 = run_indicators(df)
    AAPL = map_values(df2)
    show(AAPL)


def run_ticker(ticker: str):
    try:
        df = daily_df(ticker)
        df2 = run_indicators(df)
        df3 = map_values(df2)
        params = {str(locals()['ticker']): df3}
        show(**params)
    except:
        messagebox.showerror("Error!", "Invalid ticker entered.")


# def get_graph(ticker: str):


top = tkinter.Tk()
top.title("Technical Analysis Indicators")
top.width = "15c"
label = tkinter.Label(
    top, text="Enter a ticker to retrieve that stock's data and buy/sell indicator status.")
label.pack()
b = tkinter.Button(top, text="Run sample data", command=run_sample)
val = tkinter.Entry(top)
b.pack()
val.pack()
databutton = tkinter.Button(
    top, text="Get data for this stock", command=lambda: run_ticker(val.get().upper()))
databutton.pack()
indicatorbutton = tkinter.Button(
    top, text="Get the summarized indicator chart for this stock", command=lambda: get_graph(val2.get().upper()))
top.mainloop()
