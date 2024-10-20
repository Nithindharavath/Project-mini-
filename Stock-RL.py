import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):  # Change _init to __init__
        super(DQN, self).__init__()  # Change _init to __init__() 
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
    net_worth_history = [initial_investment]

    for episode in range(num_episodes):
        state = get_state(stocks_test, 0)
        total_reward = 0
        num_stocks = 0
        net_worth = initial_investment

        for t in range(len(stocks_test) - 1):
            action = next_act(state, epsilon, output_dim)
            next_state = get_state(stocks_test, t + 1)
            reward = 0
            done = False  # Initialize done

            close_price = stocks_test['close'].iloc[t]
            if action == 0:  # Buy
                num_stocks += 1
                net_worth -= close_price
                reward = -close_price  # Penalize for buying
            elif action == 1:  # Sell
                if num_stocks > 0:  # Only sell if we own stocks
                    num_stocks -= 1
                    net_worth += close_price
                    # Calculate profit/loss considering the previous close price
                    reward = close_price - stocks_test['close'].iloc[t-1]

            if num_stocks < 0:
                num_stocks = 0

            # Adjust net worth based on current holdings
            current_value = net_worth + (num_stocks * stocks_test['close'].iloc[t])
            total_reward += reward
            remember(state, action, reward, next_state, done)

            # Update the net worth for the history
            net_worth_history.append(current_value)
            state = next_state

            # Mark the episode as done at the last time step
            if t == len(stocks_test) - 2:
                done = True

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        replay()
        if episode % update_target_every == 0:
            update_target_network()

    return net_worth_history


def plot_net_worth(net_worth, stock_df):
    net_worth_df = pd.DataFrame(net_worth, columns=['value'])
    
    # Ensure there are enough dates for plotting
    if len(stock_df) > len(net_worth_df):
        stock_df = stock_df.iloc[:len(net_worth_df)]  # Align with the length of the net worth history

    # Plot the portfolio value over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df['date'], y=net_worth_df['value'], mode='lines', 
                             name='Portfolio Value', line=dict(color='cyan', width=2)))
    fig.update_layout(title='Change in Portfolio Value Day by Day', 
                      xaxis_title='Date', yaxis_title='Portfolio Value ($)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the start and end portfolio values
    start_net_worth = net_worth[0]  # Starting portfolio value
    end_net_worth = net_worth[-1]   # Final portfolio value
    
    st.write(f"Start Portfolio Value: {start_net_worth:.2f}")
    st.write(f"End Portfolio Value: {end_net_worth:.2f}")
    
    # Display a note based on net worth increase or decrease
    if end_net_worth > start_net_worth:
        st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> '
                    'Increase in your net worth as a result of model decisions.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> '
                    'Decrease in your net worth as a result of model decisions.</p>', unsafe_allow_html=True)


# Function to plot net worth with a dynamic note
def plot_net_worth(net_worth, stock_df):
    net_worth_df = pd.DataFrame(net_worth, columns=['value'])
    
    # Ensure there are enough dates for plotting
    if len(stock_df) > len(net_worth_df):
        stock_df = stock_df.iloc[:len(net_worth_df)]  # Align with the length of the net worth history

    # Plot the portfolio value over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df['date'], y=net_worth_df['value'], mode='lines', 
                             name='Portfolio Value', line=dict(color='cyan', width=2)))
    fig.update_layout(title='Change in Portfolio Value Day by Day', 
                      xaxis_title='Date', yaxis_title='Portfolio Value ($)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the start and end portfolio values
    start_net_worth = net_worth[0]  # Starting portfolio value
    end_net_worth = net_worth[-1]   # Final portfolio value
    
    st.write(f"<span style='color:gray;'>Start Portfolio Value: {start_net_worth:.2f}</span>", unsafe_allow_html=True)
    st.write(f"<span style='color:gray;'>End Portfolio Value: {end_net_worth:.2f}</span>", unsafe_allow_html=True)
    
    # Display a note based on net worth increase or decrease
    if end_net_worth > start_net_worth:
        note_color = "green"  # Color for increase
        note_text = "Increase in your net worth as a result of model decisions."
    else:
        note_color = "red"  # Color for decrease
        note_text = "Decrease in your net worth as a result of model decisions."
    
    st.markdown(f"<span style='color:{note_color};'>{note_text}</span>", unsafe_allow_html=True)


# User input for stock ticker, initial investment, and number of episodes
ticker = st.text_input('Enter stock ticker:', 'AAPL')
initial_investment = st.number_input('Initial investment:', min_value=100, step=100, value=1000)
num_episodes = st.number_input('Number of episodes:', min_value=1, step=1, value=50)

# Simulate stock trading
# Placeholder for stock data - Replace with actual stock data retrieval
stocks_data = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=200),
    'close': np.random.randn(200).cumsum() + 100  # Simulated stock prices
})

stock_data = data_prep(stocks_data, ticker)
net_worth_history = test_stock(stock_data, initial_investment, num_episodes)

# Plot the net worth graph
plot_net_worth(net_worth_history, stock_data)
