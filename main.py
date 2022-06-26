import copy
import datetime

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pypfopt
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import plotting

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
    start, end = get_dates()
    mu, cov, data = get_data(start, end, stocks)
    riskfree = rf()
    short, const = short_position()
    if const == 'No' and short == 'Yes':
        ef_no_bounds(riskfree, mu, cov, short, data)
    elif const == 'No' and short == 'No':
        ef_no_bounds(riskfree, mu, cov, short, data)
    elif short == "Yes" and const == 'Yes':
        ub, lb = constraints(stocks)
        ef(riskfree, mu, cov, lb, ub)
    else:
        ub = constraints_no_shorting(stocks)
        ef(riskfree, mu, cov, constrains_upper=ub)


# get stock data, return the expected mean and covariance matrix
def get_data(start, end, stocks):
    st.write(stocks)
    data = yf.download(stocks, start=start, end=end)
    data2 = data['Adj Close']
    data2.index = data2.index.strftime('%m/%d/%Y')
    st.dataframe(data2)
    mu = expected_returns.mean_historical_return(data2)
    mu = mu.sort_values(ascending=False)
    st.write('Expected Returns')
    st.dataframe(mu)
    cov = risk_models.sample_cov(data2)
    st.write('Covariance Matrix')
    st.dataframe(cov)
    st.pyplot(fig=plot_returns(data2))
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
        st.write(
            '** To ensure that the sum of weights equals 1 the minimum upper bound weight is 1/ # of stocks which is ',
            min)
        with col1:
            st.header('Upper Bound')
            upper_bound.append(
                st.number_input('Enter maximum weight that a stock can have', value=.40, min_value=min, max_value=1.0,
                                step=.01))
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
        st.write('** To ensure that the sum of weights equals 1 the minimum weight is 1/ # of stocks which is ', min)
        st.header('Upper Bound')
        upper_bound.append(
            st.number_input('Enter maximum weight that a stock can have', value=.50, min_value=min, max_value=1.0,
                            step=.01))
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
    g = ef_constraints_plt(mu, cov, riskfree, lower_constraints, constrains_upper)
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
    st.pyplot(g)


def ef_no_bounds(riskfree, mu, cov, short, data):
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
        y = pd.DataFrame(ef2.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(y)
    with col6:
        z = pd.DataFrame(ef3.portfolio_performance(risk_free_rate=riskfree, verbose=True),
                         index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        st.write(z)
    g = ef_plt(mu, cov, riskfree, weights)
    st.pyplot(g)


def plot_returns(data):
    plt.figure(figsize=(14, 7))
    for c in data.columns.values:
        sn.lineplot(x=data.index, y=data[c], data=data)
    plt.ylabel('Price in $')


def get_dates():
    with st.form('Dates'):
        start = st.date_input("start date", datetime.date(2020, 1, 1))
        end = st.date_input("end date", datetime.date(2021, 1, 1))
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write(start)
            st.write(end)
    return start, end


def ef_plt(mu, cov, riskfree, weights):
    ef = EfficientFrontier(mu, cov, weight_bounds=weights)
    fig, ax = plt.subplots()
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef ,ef_param='utility',ax=ax, show_assets=True,show_tickers=True ,zorder=5)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe(riskfree)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="orange", label="Max Sharpe",zorder=10)

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="cool",zorder=0)

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.show()


def ef_constraints_plt(mu, cov, riskfree, lower_constraints=None, constrains_upper=None):
    if lower_constraints is not None:
        weight_bounds = (-1, 1)
    else:
        weight_bounds = (0, 1)
    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef.add_constraint(lambda y: y <= constrains_upper)
    if lower_constraints is not None:
        ef.add_constraint(lambda z: z >= lower_constraints)
    fig, ax = plt.subplots()
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False, zorder=5)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe(riskfree)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="orange", label="Max Sharpe", zorder=10)

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="cool", zorder=0)

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.show()


if len(stocks) > 0:
    app(stocks)
else:
    app(default)
