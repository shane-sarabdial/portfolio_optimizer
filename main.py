import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pypfopt
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import risk_models
import seaborn as sn
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Portfolio')
default = 'SPY,AAPL,TSLA'
stocks = st.text_input('Enter up to 10 tickers seperated by commas')
if len(stocks.split(sep=',')) > 10:
    raise Exception(st.write("Sorry too many tickers"))
stocks = stocks.upper()


def app(stocks):
    mu, cov, data = get_data(stocks)
    riskfree = rf()
    short, const = short_position()
    if const == 'No' and short == 'Yes':
        ef_no_bounds(riskfree, mu, cov, short)
    elif const == 'No' and short == 'No':
        ef_no_bounds(riskfree, mu, cov, short)
    elif short == "Yes" and const == 'Yes':
        ub, lb = constraints(stocks)
        ef(riskfree, mu, cov, lb, ub)
    else:
        ub = constraints_no_shorting(stocks)
        ef(riskfree, mu, cov, constrains_upper=ub)


# get stock data, return the expected mean and covariance matrix
def get_data(stocks):
    st.write(stocks)
    data = yf.download(stocks, start='2020-01-01', end='2021-12-31')
    data2 = data['Adj Close']
    data2.index = data2.index.strftime('%m/%d/%Y')
    st.dataframe(data2)
    mu = pypfopt.expected_returns.mean_historical_return(data2)
    mu = mu.sort_values(ascending=False)
    st.dataframe(mu)
    cov = pypfopt.risk_models.sample_cov(data2)
    st.dataframe(cov)
    st.pyplot(fig =plot_returns(data2))
    st.line_chart(data2)
    return mu, cov, data2


# ask user if they want to short and add constraints
def short_position():
    short = st.radio('Do you want short positons in your portfolio?', ('Yes', 'No'), index=1)
    const = st.radio('Do you want to add constraints?', ('Yes', 'No'), index=1)
    return short, const


def constraints(stocks):
    upper_bound = []
    col1, col2 = st.columns(2)
    y = stocks.split(',')
    min = 1 / (len(y))
    with st.form('myform'):
        with col1:
            st.header('Upper Bound')
            upper_bound.append(
                st.number_input('Enter maximum weight that a stock can have', value=.30, min_value=min, max_value=1.0,
                                step=.01))
            st.write(
                '** To ensure that the sum of weights equals 1 the minimum upper bound weight is 1/ # of stocks which is ',
                min)
        lower_bound = []
        with col2:
            st.header('Lower Bound')
            lower_bound.append(
                st.number_input('Enter minimum weight that a stock can have', value=-.30, min_value=-1.0, max_value=0.0,
                                step=.01))
        submitted = st.form_submit_button("Submit")
        if submitted:
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
    return upper_bound, lower_bound


def constraints_no_shorting(stocks):
    upper_bound = []
    y = stocks.split(',')
    min = 1 / (len(y))
    with st.form('myform'):
        st.header('Upper Bound')
        upper_bound.append(
            st.number_input('Enter maximum weight that a stock can have', value=.50, min_value=min, max_value=1.0,
                            step=.01))
        st.write('** To ensure that the sum of weights equals 1 the minimum weight is 1/ # of stocks which is ', min)
        # for x in y:
        #     keys.append(x)
        #     lower_bound.append(st.number_input(f'{x}', value=0.0, min_value=0.0, max_value=1.0, step=.01))
        # for keys, upper_bound in zip(keys, lower_bound):
        #     results_upper[keys] = upper_bound
        submitted = st.form_submit_button("Submit")
        if submitted:
            lower_bound = np.array(upper_bound)
    return upper_bound


def rf():
    st.header('Risk free rate')
    st.write('Enter a risk free rate')
    rf = st.number_input("Risk Free Rate", value=0.02, min_value=0.0, max_value=2.0, step=0.01)
    return rf


def ef(riskfree, mu, cov, lower_constraints=None, constrains_upper=None):
    if lower_constraints is not None:
        weight_bounds = (-1, 1)
    else:
        weight_bounds = (0, 1)
    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    # for t in constrains_upper:
    #     if t == 0:
    #         ef.add_constraint(lambda x: x >= 0)
    #     else:
    #         ef.add_constraint(lambda y: y <= t)
    ef.add_constraint(lambda y: y <= constrains_upper)
    if lower_constraints is not None:
        ef.add_constraint(lambda z: z >= lower_constraints)
    sharpe = ef.max_sharpe(risk_free_rate=riskfree)
    clean_weights = ef.clean_weights()
    ef2 = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef2.add_constraint(lambda y: y <= constrains_upper)
    if lower_constraints is not None:
        ef2.add_constraint(lambda z: z >= lower_constraints)
    risk = ef2.min_volatility()
    clean_weights1 = ef2.clean_weights()
    ef3 = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    if lower_constraints is not None:
        ef3.add_constraint(lambda z: z >= lower_constraints)
    ef3.add_constraint(lambda y: y <= constrains_upper)
    mean = ef3._max_return()
    clean_weights2 = ef3.clean_weights()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Max Sharpe Portfolio')
        st.write(pd.DataFrame(list(clean_weights.items()), columns=['Stock', 'Weights']))
    with col2:
        st.write('Minimum Volatility Portfolio')
        st.write(pd.DataFrame(list(clean_weights1.items()), columns=['Stock', 'Weights']))
    with col3:
        st.write('Maximum Return Portfolio')
        st.write(pd.DataFrame(list(clean_weights2.items()), columns=['Stock', 'Weights']))
    col4, col5, col6 = st.columns(3)
    with col4:
        x = pd.DataFrame(ef.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(x)
    with col5:
        x = pd.DataFrame(ef2.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(x)
    with col6:
        x = pd.DataFrame(ef3.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(x)


def ef_no_bounds(riskfree, mu, cov, short):
    if short == "Yes":
        weights = (-1, 1)
    else:
        weights = (0, 1)
    ef = EfficientFrontier(mu, cov, weight_bounds=weights)
    sharpe = ef.max_sharpe(risk_free_rate=riskfree)
    clean_weights = ef.clean_weights()
    ef2 = EfficientFrontier(mu, cov, weight_bounds=weights)
    risk = ef2.min_volatility()
    clean_weights1 = ef2.clean_weights()
    ef3 = EfficientFrontier(mu, cov, weight_bounds=weights)
    mean = ef3._max_return()
    clean_weights2 = ef3.clean_weights()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Max Sharpe Portfolio')
        st.write(pd.DataFrame(list(clean_weights.items()), columns=['Stock', 'Weights']))
    with col2:
        st.write('Minimum Volatility Portfolio')
        st.write(pd.DataFrame(list(clean_weights1.items()), columns=['Stock', 'Weights']))
    with col3:
        st.write('Maximum Return Portfolio')
        st.write(pd.DataFrame(list(clean_weights2.items()), columns=['Stock', 'Weights']))
    col4, col5, col6 = st.columns(3)
    with col4:
        x = pd.DataFrame(ef.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(x)
    with col5:
        x = pd.DataFrame(ef2.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(x)
    with col6:
        x = pd.DataFrame(ef3.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(x)


def plot_returns(data):
    plt.figure(figsize=(14,7))
    for c in data.columns.values:
        sn.lineplot(x = data.index, y = data[c], data = data)
    plt.legend(loc ='upper left', fontsize = 12)
    plt.ylabel('Price in $')
    plt.xticks(data.index)




if len(stocks) > 0:
    app(stocks)
else:
    app(default)





















