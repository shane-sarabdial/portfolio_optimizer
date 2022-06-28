import copy
import datetime
import time
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pypfopt
from cvxpy import SolverError
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import plotting
from pypfopt import risk_models
import seaborn as sn
import matplotlib.pyplot as plt
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Efficient Frontier Portfolio')
st.subheader('Created by Shane Sarabdial')
st.subheader('Contact : Satramsarabdial12@gmail.com')
st.write('App is in beta. Some elements may not load the first time, try reloading the page')
default = 'AMD,NFLX,AMZN,JPM,GE'
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
        ef(riskfree, mu, cov, data, lb, ub)
    else:
        ub = constraints_no_shorting(stocks)
        ef(riskfree, mu, cov, data, constrains_upper=ub)
    st.video("https://www.youtube.com/watch?v=PiXrLGMZr1g&t=1s&ab_channel=Investopedia")

# get stock data, return the expected mean and covariance matrix

def get_data(start, end, stocks):
    st.write(stocks)
    data = yf.download(stocks, start=start, end=end)
    data2 = data['Adj Close']
    data2.index = data2.index.strftime('%m/%d/%Y')
    st.dataframe(data2)
    csv = convert_df(data2)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='stock_prices.csv',
        mime='text/csv',
    )
    st.pyplot(fig=plot_returns(data2))
    st.pyplot(plot_returns_change(data2))
    mu = pypfopt.expected_returns.mean_historical_return(data2, compounding=False)
    mu = mu.sort_values(ascending=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('#### Annual Returns')
        st.dataframe(mu)
    with col2:
        cov = risk_models.sample_cov(data2)
        st.markdown('#### Covariance Matrix')
        st.dataframe(cov)
    with st.spinner('loading...'):
        time.sleep(3)
    st.header('Efficient Frontier')
    return mu, cov, data2


# ask user if they want to short and add constraints
def short_position():
    short = st.radio('Do you want short positons in your portfolio?', ('Yes', 'No'), index=1, help='A short, or a '
                                                                                                   'short position, '
                                                                                                   'is created when a '
                                                                                                   'trader sells a '
                                                                                                   'security first '
                                                                                                   'with the '
                                                                                                   'intention of '
                                                                                                   'repurchasing it '
                                                                                                   'or covering it '
                                                                                                   'later at a lower '
                                                                                                   'price. A trader '
                                                                                                   'may decide to '
                                                                                                   'short a security '
                                                                                                   'when she believes '
                                                                                                   'that the price of '
                                                                                                   'that security is '
                                                                                                   'likely to '
                                                                                                   'decrease in the '
                                                                                                   'near future.')
    const = st.radio('Do you want to add constraints?', ('Yes', 'No'), index=0, help='Adding constraints will place '
                                                                                     'restrictions on the maximum '
                                                                                     'weights that any 1 long or '
                                                                                     'short postion can have '
                     )
    return short, const


def target():
    none = st.radio('Do you have a target?', 'No,return,volatility,sharpe', index=0)
    return none


def constraints(stocks):
    upper_bound = []
    lower_bound = []
    col1, col2 = st.columns(2)
    y = stocks.split(',')
    min = 1 / (len(y))
    with st.form('myform'):
        st.warning('The solver may not be able to solve for the constraints given, if error occurs try changing the '
                   'weights')
        with col1:
            upper_bound.append(
                st.number_input('Upper Bound', value=.60, min_value=min, max_value=1.0,
                                step=.01,
                                help=f"Set the maximum weight any 1 stock can have. To ensure that the sum of "
                                     f"weights equals 1, the minimum upper bound weight is 1/ # of stocks which"
                                     f" is  {min}"))

        with col2:
            lower_bound.append(
                st.number_input('Lower Bound', value=-.30, min_value=-1.0, max_value=0.0,
                                step=.01, help='Set the maximum weight that any 1 stock can be shorted'))
        submitted = st.form_submit_button("Submit", help=(f"To ensure that the sum of weights equals 1, the minimum "
                                                          f"upper bound weight is 1/ # of stocks which is  {min}"))
    if submitted:
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
    return upper_bound, lower_bound


def constraints_no_shorting(stocks):
    upper_bound = []
    y = stocks.split(',')
    min = 1 / (len(y))
    with st.form('myform'):
        upper_bound.append(
            st.number_input('Enter maximum weight that a stock can have', value=.50, min_value=min, max_value=1.0,
                            step=.01, help=f"Set the maximum weight any 1 stock can have. To ensure that the sum of "
                                           f"weights equals 1, the minimum upper bound weight is 1/ # of stocks which"
                                           f" is  {min}"))
        # for x in y:
        #     keys.append(x)
        #     lower_bound.append(st.number_input(f'{x}', value=0.0, min_value=0.0, max_value=1.0, step=.01))
        # for keys, upper_bound in zip(keys, lower_bound):
        #     results_upper[keys] = upper_bound
        st.warning('The solver may not be able to solve for the constraints given, if error occurs try changing the '
                   'weights')
        submitted = st.form_submit_button("Submit")
        if submitted:
            lower_bound = np.array(upper_bound)
    return upper_bound


def rf():
    rf = st.number_input("Risk Free Rate", value=0.02, min_value=0.0, max_value=2.0, step=0.01, help="The risk-free "
                                                                                                     "rate of return "
                                                                                                     "is the "
                                                                                                     "theoretical "
                                                                                                     "rate of return "
                                                                                                     "of an "
                                                                                                     "investment with "
                                                                                                     "zero risk. "
                                                                                                     "Typically "
                                                                                                     "treasury bills "
                                                                                                     "are used as the "
                                                                                                     "risk free "
                                                                                                     "rate.")
    return rf


def ef(riskfree, mu, cov, data, lower_constraints=None, constrains_upper=None):
    port_val = st.number_input('Enter the value of your portfolio', value=10000.0, help='The share price used for '
                                                                                        'allocation of shares is the '
                                                                                        'last price in the dataset. To '
                                                                                        'get a accurate share '
                                                                                        'allocation change the end '
                                                                                        'date to today or yesterdays '
                                                                                        'date.')
    if lower_constraints is not None:
        weight_bounds = (-1, 1)
    else:
        weight_bounds = (0, 1)
    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef.add_constraint(lambda y: y <= constrains_upper)
    if lower_constraints is not None:
        ef.add_constraint(lambda z: z >= lower_constraints)
    g = ef_constraints_plt(mu, cov, riskfree, lower_constraints, constrains_upper)
    sharpe = ef.max_sharpe(risk_free_rate=riskfree)
    clean_weights = ef.clean_weights()
    all, lo = allocation(data, clean_weights, port_val)
    ef2 = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef2.add_constraint(lambda y: y <= constrains_upper)
    if lower_constraints is not None:
        ef2.add_constraint(lambda z: z >= lower_constraints)
    risk = ef2.min_volatility()
    clean_weights1 = ef2.clean_weights()
    all1, lo1 = allocation(data, clean_weights1, port_val)
    ef3 = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    if lower_constraints is not None:
        ef3.add_constraint(lambda z: z >= lower_constraints)
    ef3.add_constraint(lambda y: y <= constrains_upper)
    mean = ef3._max_return()
    clean_weights2 = ef3.clean_weights()
    all2, lo2 = allocation(data, clean_weights2, port_val)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Max Sharpe Portfolio')
        df1 = pd.DataFrame(list(clean_weights.items()), columns=['Stock', 'Weights'])
        df1 = df1.merge(all, on='Stock', how='outer')
        df1.fillna(0, inplace=True)
        df1['Shares'] = df1['Shares'].astype('int')
        st.write(df1)
        st.write('Funds remaining: ${:.2f}'.format(lo))
    with col2:
        st.write('Minimum Volatility Portfolio')
        df2 = pd.DataFrame(list(clean_weights1.items()), columns=['Stock', 'Weights'])
        df2 = df2.merge(all1, on='Stock', how='outer')
        df2.fillna(0, inplace=True)
        df2['Shares'] = df2['Shares'].astype('int')
        st.write(df2)
        st.write('Funds remaining: ${:.2f}'.format(lo1))
    with col3:
        st.write('Maximum Return Portfolio')
        df3 = pd.DataFrame(list(clean_weights2.items()), columns=['Stock', 'Weights'])
        df3 = df3.merge(all2, on='Stock', how='outer')
        df3.fillna(0, inplace=True)
        df3['Shares'] = df3['Shares'].astype('int')
        st.write(df3)
        st.write('Funds remaining: ${:.2f}'.format(lo2))
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
    st.caption('Any Portfolio above the efficient frontier cannot exist')


def ef_no_bounds(riskfree, mu, cov, short, data):
    port_val = st.number_input('Enter the value of your portfolio', value=10000.0, help='The share price used for '
                                                                                        'allocation of shares is the '
                                                                                        'last price in the dataset. To '
                                                                                        'get a accurate share '
                                                                                        'allocation change the end '
                                                                                        'date to today or yesterdays '
                                                                                        'date.')
    if short == "Yes":
        weights = (-1, 1)
    else:
        weights = (0, 1)
    ef = EfficientFrontier(mu, cov, weight_bounds=weights)
    sharpe = ef.max_sharpe(risk_free_rate=riskfree)
    clean_weights = ef.clean_weights()
    all, lo = allocation(data, clean_weights, port_val)
    ef2 = EfficientFrontier(mu, cov, weight_bounds=weights)
    risk = ef2.min_volatility()
    clean_weights1 = ef2.clean_weights()
    all1, lo1 = allocation(data, clean_weights1, port_val)
    ef3 = EfficientFrontier(mu, cov, weight_bounds=weights)
    mean = ef3._max_return()
    clean_weights2 = ef3.clean_weights()
    all2, lo2 = allocation(data, clean_weights2, port_val)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Max Sharpe Portfolio')
        df1 = pd.DataFrame(list(clean_weights.items()), columns=['Stock', 'Weights'])
        df1 = df1.merge(all, on='Stock', how='outer')
        df1.fillna(0, inplace=True)
        df1['Shares'] = df1['Shares'].astype('int')
        st.write(df1)
        st.write('Funds remaining: ${:.2f}'.format(lo))
    with col2:
        st.write('Minimum Volatility Portfolio')
        df2 = pd.DataFrame(list(clean_weights1.items()), columns=['Stock', 'Weights'])
        df2 = df2.merge(all1, on='Stock', how='outer')
        df2.fillna(0, inplace=True)
        df2['Shares'] = df2['Shares'].astype('int')
        st.write(df2)
        st.write('Funds remaining: ${:.2f}'.format(lo1))
    with col3:
        st.write('Maximum Return Portfolio')
        df3 = pd.DataFrame(list(clean_weights2.items()), columns=['Stock', 'Weights'])
        df3 = df3.merge(all2, on='Stock', how='outer')
        df3.fillna(0, inplace=True)
        df3['Shares'] = df3['Shares'].astype('int')
        st.write(df3)
        st.write('Funds remaining: ${:.2f}'.format(lo2))
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
    st.caption('Any Portfolio above the efficient frontier cannot exist')


@st.experimental_memo
def plot_returns(data):
    plt.figure(figsize=(14, 7))
    for c in data.columns.values:
        sn.lineplot(x=data.index, y=data[c], data=data, label=c)
    plt.ylabel('Price in $', fontsize=20)
    plt.xlabel('Date ', fontsize=20)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(0, len(data.index), step=60), rotation=-75)
    plt.title('Daily Returns', fontsize=20)


@st.experimental_memo
def plot_returns_change(data):
    plt.figure(figsize=(14, 7))
    data1 = data.pct_change()
    data1.dropna(inplace=True)
    for c in data1.columns.values:
        sn.lineplot(x=data1.index, y=data1[c], data=data1, label=c)
    plt.ylabel('% Change', fontsize=20)
    plt.xlabel('Date ', fontsize=20)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(0, len(data1.index), step=60), rotation=-75)
    plt.title('Daily Returns % Change', fontsize=20)


def get_dates():
    with st.form('Dates'):
        start = st.date_input("start date", datetime.date(2016, 1, 1))
        end = st.date_input("end date", datetime.date(2021, 12, 31))
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write(start)
            st.write(end)
    return start, end


@st.experimental_memo(suppress_st_warning=True)
def ef_plt(mu, cov, riskfree, weights):
    ef = EfficientFrontier(mu, cov, weight_bounds=(None, None))
    fig, ax = plt.subplots()
    ef_max_sharpe = copy.deepcopy(ef)
    ef_min_vol = copy.deepcopy(ef)
    try:
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False, zorder=5)
    except SolverError:
        st.markdown('### Solver error, try removing a symbol or changing weights')

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe(riskfree)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance(verbose=True)
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="orange", label="Max Sharpe", zorder=20)

    ef_min_vol.min_volatility()
    ret_tangent_vol, std_tangent_vol, _ = ef_min_vol.portfolio_performance()
    ax.scatter(std_tangent_vol, ret_tangent_vol, marker="*", s=100, c="#cc2975", label="Minimum Volatility", zorder=20)

    # Generate random portfolios
    n_samples = 8000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="cool", zorder=0)

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()


@st.experimental_memo(suppress_st_warning=True)
def ef_constraints_plt(mu, cov, riskfree, lower_constraints=None, constrains_upper=None):
    ef = EfficientFrontier(mu, cov, weight_bounds=(None, None))
    ef.add_constraint(lambda y: y <= constrains_upper)
    if lower_constraints is not None:
        ef.add_constraint(lambda z: z >= lower_constraints)
    fig, ax = plt.subplots()
    ef_max_sharpe = copy.deepcopy(ef)
    ef_min_vol = copy.deepcopy(ef)
    try:
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False, zorder=5)
    except SolverError:
        st.markdown('### Solver error, try removing a symbol or changing weights')
    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe(riskfree)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="orange", label="Max Sharpe", zorder=10)

    ef_min_vol.min_volatility()
    ret_tangent_vol, std_tangent_vol, _ = ef_min_vol.portfolio_performance()
    ax.scatter(std_tangent_vol, ret_tangent_vol, marker="*", s=100, c="#cc2975", label="Minimum Volatility", zorder=20)

    # Generate random portfolios
    n_samples = 8000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="cool", zorder=0)

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()




@st.experimental_memo
def convert_df(data):
    return data.to_csv().encode('utf-8')


def allocation(data, weights, port_val):
    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(weights, latest_prices, port_val)
    try:
        allocation, leftover = da.greedy_portfolio()
    except:
        allocation, leftover = da.lp_portfolio()
    allocation = pd.DataFrame(list(allocation.items()), columns=['Stock', 'Shares'])
    return allocation, leftover


if len(stocks) > 0:
    app(stocks)
else:
    app(default)
