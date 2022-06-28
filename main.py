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
from datetime import timedelta, date

# set page config
st.set_page_config(page_title='Portfolio Optimizer')
hide_menu_style = """
    <style>
    footer{visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Efficient Frontier Portfolio')
st.subheader('Created by Shane Sarabdial')
st.subheader('Contact : Satramsarabdial12@gmail.com')
st.write('App is in beta. Some elements may not load the first time, try reloading the page')

def get_stock():
    stocks = st.text_input('Enter up to 10 tickers seperated by commas and more than 3')
    stocks = stocks.upper()
    if len(stocks.split(sep=',')) > 10:
        raise Exception(st.write("Sorry too many tickers"))
    if len(stocks.split(sep=',')) < 3:
        st.caption('Using less than 3 stocks will cause errors, stocks defaulted to assigned tickers. Please use more '
                   'than 3 tickers ')
        stocks = 'AMD,NFLX,AMZN,JPM,GE'
    return stocks


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


# takes user input and gets stock data from yahoo
# calls function to plot data and returns/covariance dataframes
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
    st.pyplot(plot_returns(data2))
    st.pyplot(plot_returns_change(data2))
    mu, cov = ret_cov(data2)
    st.header('Efficient Frontier')
    return mu, cov, data2


# function for creating returns and covriance dataframe
@st.experimental_memo(suppress_st_warning=True)
def ret_cov(data):
    mu = pypfopt.expected_returns.mean_historical_return(data, compounding=False)
    mu = mu.sort_values(ascending=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('#### Annual Returns')
        st.dataframe(mu)
    with col2:
        cov = risk_models.sample_cov(data)
        st.markdown('#### Covariance Matrix')
        st.dataframe(cov)
    with st.spinner('loading...'):
        time.sleep(3)
    return mu, cov


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


# gets lower and upper bound constraints from user
# creates a form which the user will inout data
def constraints(stocks):
    upper_bound = []
    lower_bound = []
    y = stocks.split(',')
    min = 1 / (len(y))
    with st.form('myform'):
        st.warning('The solver may not be able to solve for the constraints given, if error occurs try changing the '
                   'weights')
        upper_bound.append(
            st.number_input('Upper Bound', value=.60, min_value=min, max_value=1.0,
                            step=.01,
                            help=f"Set the maximum weight any 1 stock can have. To ensure that the sum of "
                                 f"weights equals 1, the minimum upper bound weight is 1/ # of stocks which"
                                 f" is  {min}"))
        lower_bound.append(
            st.number_input('Lower Bound', value=-.30, min_value=-1.0, max_value=0.0,
                            step=.01, help='Set the maximum weight that any 1 stock can be shorted'))
        submitted = st.form_submit_button("Submit")
    if submitted:
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
    return upper_bound, lower_bound


# gets upper bound constraints from user
# creates a form which the user will inout data
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

# define risk free rate and gets user input
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

# in this function the user selects constraints = yes and (short = no or yes)
# checks if user wants to short and adds constraints
# takes lower and upper bound and creates constraints for the efficient frontier
# creates several dataframes that return the max sharpe, min volatility, and max return portfolios
# plots the efficient frontier
# creates a dataframe that defines share allocation based on the portfolio value
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

# in this function the user selects constraints = no and (short = no or yes)
# checks if user wants to short and adds constraints
# in this function the user selected no constraints but can short or long
# creates several dataframes that return the max sharpe, min volatility, and max return portfolios
# plots the efficient frontier
# creates a dataframe that defines share allocation based on the portfolio value
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

# function plots the returns of stocks
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

# function plots the % returns
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

# function gets dates from user and sets the default dates
def get_dates():
    today = date.today()
    yesterday = today - timedelta(days=1)
    years_ago = today - timedelta(1825)
    with st.form('Dates'):
        start = st.date_input("start date", years_ago)
        end = st.date_input("end date", yesterday)
        submitted = st.form_submit_button('Submit')
        if submitted:
            print(start, end)
    return start, end

# function plots the efficient frontier and random portfolios
# code can be found on pyportfolioopt docs
# plot is used when user has no constraints and can either short or go long
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

# function plots the efficient frontier and random portfolios
# code can be found on pyportfolioopt docs
# plot is used when user  HAS constraints and can either short or go long
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

# creates a csv file for user to download
@st.experimental_memo
def convert_df(data):
    return data.to_csv().encode('utf-8')

# uses two methods to allocate stocks based on the users portfolio value
# if greedy method fail use lp method
def allocation(data, weights, port_val):
    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(weights, latest_prices, port_val)
    try:
        allocation, leftover = da.greedy_portfolio()
    except:
        allocation, leftover = da.lp_portfolio()
    allocation = pd.DataFrame(list(allocation.items()), columns=['Stock', 'Shares'])
    return allocation, leftover

# checks if user has in
stocks = get_stock()
app(stocks)

