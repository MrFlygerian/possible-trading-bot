import yfinance as yf
import pandas as pd
import datetime
import streamlit as st
from finta import TA


@st.cache
def _get_historical_data(ticker: object, num_days: int, interval: object) -> object:
    """
    Function that uses the yfinance API to get stock data
    :rtype: object
    :return:
    """

    start = (datetime.date.today() - datetime.timedelta(num_days))
    end = datetime.datetime.today()

    data = yf.download(ticker, start=start, end=end, interval=interval)
    data.rename(columns={"Close": 'close',
                         "Adj Close": 'adj close',
                         "High": 'high',
                         "Low": 'low',
                         'Volume': 'volume',
                         'Open': 'open'},
                inplace=True)
    return data


def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """
    return data.ewm(alpha=alpha).mean()


def _get_indicator_data(data, indicators):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    Altered dataframe
    """

    for indicator in indicators:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['adj close'])

    return data


def _produce_prediction(data, window=7) -> object:
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :rtype: object
    :param window: number of days, or rows to look ahead to see what the price did
    """

    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    data.dropna(inplace=True)

    return data


@st.cache
def _get_historical_data_mt(tickers, num_days, interval):
    """
    Function that uses the yfinance API to get stock data
    :return:
    """

    start = (datetime.date.today() - datetime.timedelta(num_days))
    end = datetime.datetime.today()

    data = yf.download(tickers, start=start, end=end, interval=interval, group_by='Ticker')
    data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    data.rename(columns={"Close": 'close',
                         "Adj Close": 'adj close',
                         "High": 'high',
                         "Low": 'low',
                         'Volume': 'volume',
                         'Open': 'open'},
                inplace=True)
    return data
