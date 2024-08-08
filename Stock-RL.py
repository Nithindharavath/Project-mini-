import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define DQN Model
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

# Initialize DQN
input_dim = 3  # Number of state features
output_dim = 3  # Number of actions: Buy, Sell, Hold
dqn = DQN(input_dim, output_dim)
target_dqn = DQN(input_dim, output_dim)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters())
loss_fn = nn.MSELoss()

# Hyperparameters
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
epsilon = epsilon_start
memory = deque(maxlen=10000)
batch_size = 64
update_target_every = 10

# Cache the data preparation function
@st.cache_data
def data_prep(data, name):
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    df['5day_MA'][:4] = 0
    return df

# Cache the state representation function
def get_state(data, t):
    long_ma = data['5day_MA'].iloc[t]
    short_ma = data['1day_MA'].iloc[t]
    cash_in_hand = 1 if t == 1 else 0
    return np.array([long_ma, short_ma, cash_in_hand])

# Experience Replay
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def next_act(state, epsilon, action_dim):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        return torch.argmax(q_values).item()

def replay():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_dqn(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target_network():
    target_dqn.load_state_dict(dqn.state_dict())

def trade_t(num_of_stocks, port_value, current_price):
    return 1 if port_value > current_price else 0

def test_stock(stocks_test, initial_investment, num_episodes):
    global epsilon
    net_worth_history = []
    
    for episode in range(num_episodes):
        state = get_state(stocks_test, 0)
        total_reward = 0
        num_stocks = 0
        net_worth = initial_investment

        for t in range(len(stocks_test) - 1):
            action = next_act(state, epsilon, output_dim)
            next_state = get_state(stocks_test, t + 1)
            reward = 0

            close_price = stocks_test['close'].iloc[t]
            if action == 0:  # Buy
                num_stocks += 1
                net_worth -= close_price
                reward = -close_price
            elif action == 1:  # Sell
                num_stocks -= 1
                net_worth += close_price
                reward = close_price

            if num_stocks < 0:
                num_stocks = 0

            done = t == len(stocks_test) - 2
            total_reward += reward
            remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        replay()
        if episode % update_target_every == 0:
            update_target_network()

        net_worth_history.append(net_worth)
    
    return net_worth_history

# Function to plot net worth
def plot_net_worth(net_worth):
    net_worth_df = pd.DataFrame(net_worth, columns=['value'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net_worth_df.index, y=net_worth_df['value'], mode='lines', name='Portfolio Value', line=dict(color='cyan', width=2)))
    fig.update_layout(title='Change in Portfolio Value Day by Day', xaxis_title='Number of Days since Feb 2013', yaxis_title='Value ($)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> Increase in your net worth as a result of a model decision.</p>', unsafe_allow_html=True)

# Function to calculate performance metrics
def calculate_performance_metrics(net_worth, initial_investment):
    net_worth = np.array(net_worth)
    returns = (net_worth[-1] - initial_investment) / initial_investment
    annualized_return = (net_worth[-1] / initial_investment) ** (365 / len(net_worth)) - 1
    daily_returns = np.diff(net_worth) / net_worth[:-1]
    volatility = np.std(daily_returns)
    sharpe_ratio = annualized_return / volatility

    return {
        "Total Return": returns,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio
    }

# Function to display performance metrics
def display_performance_metrics(metrics):
    st.write("### Performance Metrics")
    for key, value in metrics.items():
        st.write(f"**{key}:** {value:.2f}")

# Function to display the table of stock movements
def show_stock_movements(data):
    data['Movement'] = data.groupby('Name')['close'].transform(lambda x: x.pct_change()).fillna(0)
    data['Trend'] = data['Movement'].apply(lambda x: 'Upward' if x > 0 else 'Downward')
    trend_summary = data.groupby(['Name', 'Trend']).size().reset_index(name='Count')
    trend_summary_pivot = trend_summary.pivot(index='Name', columns='Trend', values='Count').fillna(0).reset_index()
    st.write("### Stock Movements Summary")
    st.dataframe(trend_summary_pivot)

def main():
    st.title("Optimizing Stock Trading Strategy With Reinforcement Learning")
    
    tabs = ["Home", "Data Exploration", "Strategy Simulation", "Performance Metrics"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home":
        data = pd.read_csv('all_stocks_5yr.csv')
        show_stock_movements(data)
    
    elif selected_tab == "Data Exploration":
        data_exploration()
    
    elif selected_tab == "Strategy Simulation":
        strategy_simulation()
    
    elif selected_tab == "Performance Metrics":
        performance_metrics()

def data_exploration():
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")
    
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        show_stock_trend(stock, stock_df)

def show_stock_trend(stock, stock_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Stock Trend', line=dict(color='cyan', width=2)))
    fig.update_layout(title='Stock Trend of ' + stock, xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig, use_container_width=True)

def strategy_simulation():
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")
    
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        num_episodes = st.sidebar.slider("Number of Episodes", min_value=1, max_value=100, value=10)
        initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, value=10000)
        
        if st.sidebar.button("Run Simulation"):
            with st.spinner("Running Simulation..."):
                net_worth = test_stock(stock_df, initial_investment, num_episodes)
                plot_net_worth(net_worth)
                metrics = calculate_performance_metrics(net_worth, initial_investment)
                display_performance_metrics(metrics)

def performance_metrics():
    # Placeholder function for performance metrics tab
    st.write("### Performance Metrics")
    st.write("Performance metrics will be displayed here after simulation.")

if __name__ == "__main__":
    main()




