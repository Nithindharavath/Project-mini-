import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define DQN Model (unchanged)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize DQN (unchanged)
input_dim = 3  # Number of state features
output_dim = 3  # Number of actions: Buy, Sell, Hold
dqn = DQN(input_dim, output_dim)
target_dqn = DQN(input_dim, output_dim)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters())
loss_fn = nn.MSELoss()

# Hyperparameters (unchanged)
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
epsilon = epsilon_start
memory = deque(maxlen=10000)
batch_size = 64
update_target_every = 10

# Cache the data preparation function (unchanged)
@st.cache_data
def data_prep(data, name):
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    df['5day_MA'][:4] = 0
    return df

# Add function to prepare data for multiple stocks for comparison
def data_prep_for_comparison(data, selected_stocks, start_date, end_date):
    stock_data = {}
    for stock in selected_stocks:
        df = pd.DataFrame(data[data['Name'] == stock])
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df['5day_MA'] = df['close'].rolling(5).mean()
        stock_data[stock] = df.reset_index(drop=True)
    return stock_data

# Add function to show comparison of stock performance
def show_comparison(stock_data):
    fig = go.Figure()

    # Add stock price trends
    for stock, df in stock_data.items():
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name=f'{stock} Close Price'))
    
    fig.update_layout(title='Stock Price Trends Comparison', xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative returns and volatility comparison
    display_metrics(stock_data)

# Add function to display metrics like returns, volatility, and moving averages
def display_metrics(stock_data):
    st.write("### Key Metrics")
    
    metrics = []
    for stock, df in stock_data.items():
        initial_price = df['close'].iloc[0]
        final_price = df['close'].iloc[-1]
        total_return = (final_price - initial_price) / initial_price
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std()
        avg_5day_ma = df['5day_MA'].mean()
        
        metrics.append({
            "Stock": stock,
            "Initial Price": f"${initial_price:.2f}",
            "Final Price": f"${final_price:.2f}",
            "Total Return (%)": f"{total_return * 100:.2f}",
            "Volatility": f"{volatility:.4f}",
            "5-day MA": f"${avg_5day_ma:.2f}"
        })

    metrics_df = pd.DataFrame(metrics)
    st.write(metrics_df)

# Add function to compare moving averages
def show_moving_average_comparison(stock_data):
    fig = go.Figure()

    # Add 5-day moving averages
    for stock, df in stock_data.items():
        fig.add_trace(go.Scatter(x=df['date'], y=df['5day_MA'], mode='lines', name=f'{stock} 5-day MA'))
    
    fig.update_layout(title='5-Day Moving Averages Comparison', xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig, use_container_width=True)

# Add the compare stocks function to be included in the sidebar
def compare_stocks():
    st.header("Compare Stock Performance")
    
    # Load dataset
    data = pd.read_csv('all_stocks_5yr.csv')
    
    # List of stock names
    stock_names = list(data['Name'].unique())
    
    # User input for stocks and timeframe
    selected_stocks = st.multiselect("Choose Stocks to Compare", stock_names)
    start_date = st.date_input("Start Date", value=pd.to_datetime('2013-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2018-01-01'))
    
    if len(selected_stocks) >= 2:
        stock_data = data_prep_for_comparison(data, selected_stocks, start_date, end_date)
        show_comparison(stock_data)
        show_moving_average_comparison(stock_data)
    else:
        st.write("Please select at least two stocks to compare.")

# Define the main function
def main():
    st.title("Optimizing Stock Trading Strategy With Reinforcement Learning")
    
    tabs = ["Home", "Data Exploration", "Strategy Simulation", "Compare Stock Performance"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home":
        home_page()
    
    elif selected_tab == "Data Exploration":
        data_exploration()
    
    elif selected_tab == "Strategy Simulation":
        strategy_simulation()
    
    elif selected_tab == "Compare Stock Performance":
        compare_stocks()

# Other functions remain the same as your original code for home page, strategy simulation, etc.

if __name__ == '__main__':
    main()
