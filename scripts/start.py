import streamlit as st
import numpy as np
import datetime
import altair as alt
import yfinance as yf
import functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from multiapp_funcs import MultiApp

from sklearn.model_selection import cross_val_score

st.write('This is a streamlit app to be used for trading. This is NOT financial advice.')
app = MultiApp()

# TODO:
# Add pages to streamlit
# app.add_app("Home", home.app)
# app.add_app("Data", data.app)
# app.add_app("Model", model.app)



NUM_DAYS = 4000  # The number of days of historical data to retrieve
INTERVAL = '1d'  # Sample rate of historical data

WINDOW = st.sidebar.slider('window', 5, 14, 1)

INDICATORS = ['RSI', 'MACD',
              'STOCH', 'ADL',
              'ATR', 'MOM',
              'MFI', 'ROC',
              'OBV', 'CCI',
              'EMV', 'VORTEX']

# TODO:
# Add DL and ensemble model
MODELS = {'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1),
          'K Neighbours': KNeighborsClassifier(n_neighbors=50, n_jobs=-1)}


def clean_data(df, alpha=0.65):
    """
    Combine all data manipulation functions

    :param df: dataframe
    :param alpha: exponent factor
    :return:
    Dataframe with features and prediction target
    """
    df = functions._exponential_smooth(df, alpha=alpha)
    df = functions._get_indicator_data(df, indicators=INDICATORS)

    return df


tickers = ('TSLA', 'GOOGL',
           'GME', 'AMZN',
           'DIS', 'MSFT',
           'MA', 'AAPL')

ticker = st.sidebar.selectbox(
    "Ticker",
    tickers)

stock_df = functions._get_historical_data(ticker, NUM_DAYS, INTERVAL)
tick = yf.Ticker(ticker)

# Charting
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['Date'], empty='none')
stock_chart = alt.Chart(stock_df.reset_index()).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(title='')),
    y=alt.Y('close:Q', axis=alt.Axis(title='', format='$f')),
    tooltip=[alt.Tooltip('volume'),
             alt.Tooltip('close:Q', format='$.2f'),
             ],
).properties(
    width=900,
    height=400
).interactive().add_selection(
    nearest
)

st.altair_chart(stock_chart)

# The rest
"This is the data before being engineered"
if st.checkbox('Show dirty data'):
    st.dataframe(stock_df)

engineered_df = functions._produce_prediction(
    clean_data(stock_df), window=WINDOW
)

"This is the data after being engineered"
if st.checkbox('Show clean data'):
    st.dataframe(engineered_df)

X = engineered_df.drop("pred", axis=1)
y = engineered_df.pred

model = st.sidebar.radio(
    "Model",
    ('K Neighbours', 'Random Forest'))

f"Accuracy for {model}:"
st.write(round(np.mean(cross_val_score(MODELS[model],
                                       X, y, cv=8)), 4))

buy_sell = {1: f' ## Stock going up, buy',
            0: f'## Stock going down, sell'}

f"This is the prediction for next {WINDOW} days:"
pred_df = (clean_data(stock_df))
if datetime.datetime.today() - pred_df[-1:].index < datetime.timedelta(WINDOW):
    clf = MODELS[model]
    clf.fit(X, y)
    st.write(buy_sell[clf.predict(pred_df[-1:])[0]])
else:
    st.write('No prediction available')

if st.checkbox('Show data being predicted upon'):
    st.dataframe(pred_df[-1:])

st.write(f'Predicting for {(pred_df[-1:].index + datetime.timedelta(WINDOW)).strftime("%A, %d %b %Y")[0]}')

f'[Link to eToro for {ticker}](https://www.etoro.com/markets/{ticker.lower()})'

if st.button("Make Trade (doesn't really do anything)"):
    st.write(f'Made trade on {ticker} at {datetime.datetime.today().strftime("%A, %d %b %Y")}'
             f' set to close on {(pred_df[-1:].index + datetime.timedelta(WINDOW)).strftime("%A, %d %b %Y")[0]} ')
