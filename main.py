import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pypfopt
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import risk_models

st.header('Portfolio')
default ='SPY,AAPL,TSLA'
stocks = st.text_input('Enter up to 10 tickers seperated by commas')
if len(stocks.split(sep=',')) > 10:
    raise Exception(st.write("Sorry too many tickers"))
stocks = stocks.upper()
def home(stocks):
    st.write((stocks))
    data = yf.download(stocks, start='2020-01-01', end='2021-12-31')
    data2 = data['Adj Close']
    st.dataframe(data2)
    mu = pypfopt.expected_returns.mean_historical_return(data2)
    mu = mu.sort_values(ascending=False)
    st.dataframe(mu)
    cov = pypfopt.risk_models.sample_cov(data2)
    st.dataframe(cov)
    keys = []
    upper_bound = []
    col1, col2 = st.columns(2)
    with st.form('myform'):
        with col1:
            st.header('Upper Bound')
            for x in stocks.split(','):
                keys.append(x)
                upper_bound.append(st.number_input(f'{x}', value=0.0, min_value=0.0, max_value=1.0, step=.1))


        keys_lower =[]
        lower_bound =[]
        with col2:
            st.header('Lower Bound')
            for x in stocks.split(','):
                keys_lower.append(x)
                lower_bound.append(st.number_input(f'{x}', value=0.0, min_value=-1.0, max_value=0.0, step=.1))
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(lower_bound)




if len(stocks)>0:
    home(stocks)
else:
    home(default)