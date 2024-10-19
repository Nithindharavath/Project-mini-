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

            close_price = stocks_test['close'].iloc[t]
            if action == 0:  # Buy
                num_stocks += 1
                net_worth -= close_price
                reward = -close_price  # Penalize for buying
            elif action == 1:  # Sell
                if num_stocks > 0:  # Only sell if we own stocks
                    num_stocks -= 1
                    net_worth += close_price
                    reward = close_price  # Reward for selling

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
    
    st.write(f"Start Portfolio Value: {start_net_worth:.2f}")
    st.write(f"End Portfolio Value: {end_net_worth:.2f}")
    
    # Display a note based on net worth increase or decrease
    if end_net_worth > start_net_worth:
        st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> '
                    'Increase in your net worth as a result of model decisions.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> '
                    'Decrease in your net worth as a result of model decisions.</p>', unsafe_allow_html=True)

# Function to calculate performance metrics
def calculate_performance_metrics(net_worth, initial_investment):
    net_worth = np.array(net_worth)
    returns = (net_worth[-1] - initial_investment) / initial_investment
    annualized_return = (net_worth[-1] / initial_investment) ** (365 / len(net_worth)) - 1
    daily_returns = np.diff(net_worth) / net_worth[:-1]
    volatility = np.std(daily_returns)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0  # Prevent division by zero

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
        st.write(f"{key}: {value:.2f}")

def main():
    st.title("Enhancing Stock Trading Strategy Using Reinforcement Learning")
    
    tabs = ["Home", "Data Exploration", "Strategy Simulation"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home":
        home_page()
    
    elif selected_tab == "Data Exploration":
        data_exploration()
    
    elif selected_tab == "Strategy Simulation":
        strategy_simulation()

def home_page():
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")
    
    # Prepare to gather insights
    insights = []
    
    for name in names[1:]:
        df = data_prep(data, name)
        avg_closing_price = df['close'].mean()  # Average closing price
        initial_closing_price = df['close'].iloc[0]
        performance_trend = "Rising" if avg_closing_price > initial_closing_price else "Falling"

        # Calculate 30-day moving average for recent performance assessment
        df['30day_MA'] = df['close'].rolling(window=30).mean()
        recent_movement = "Positive" if df['30day_MA'].iloc[-1] > df['30day_MA'].iloc[-2] else "Negative"

        # Determine potential for profit based on trends and recent performance
        if performance_trend == "Rising" and recent_movement == "Positive":
            potential_profit = "Highly Favorable"
        elif performance_trend == "Falling" and recent_movement == "Negative":
            potential_profit = "Not Recommended"
        else:
            potential_profit = "Watch Closely"

        # Include volatility and max drawdown
        price_volatility = df['close'].std()
        max_drawdown = (df['close'].max() - df['close'].min()) / df['close'].max()

        insights.append({
            "Company": name,
            "Performance Trend": performance_trend,
            "Average Closing Price": avg_closing_price,
            "Potential Profit": potential_profit,
            "Price Volatility": price_volatility,
            "Max Drawdown (%)": max_drawdown * 100  # Convert to percentage
        })

    # Create a DataFrame and sort by Average Closing Price
    insights_df = pd.DataFrame(insights)
    insights_df = insights_df.sort_values(by="Average Closing Price", ascending=False)

    # Create interactive visualizations using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=insights_df['Company'], y=insights_df['Average Closing Price'], name='Avg Closing Price'))
    fig.add_trace(go.Scatter(x=insights_df['Company'], y=insights_df['Max Drawdown (%)'], mode='lines+markers', name='Max Drawdown (%)', yaxis='y2'))

    # Update layout for dual-axis chart
    fig.update_layout(
        title="Company Performance Overview",
        xaxis_title="Company",
        yaxis_title="Average Closing Price ($)",
        yaxis2=dict(title="Max Drawdown (%)", overlaying='y', side='right'),
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Insights Summary")
    for index, row in insights_df.iterrows():
        st.write(f"**{row['Company']}**: {row['Potential Profit']} based on historical trends.")
        st.write(f"- **Performance Trend**: {row['Performance Trend']}")
        st.write(f"- **Average Closing Price**: ${row['Average Closing Price']:.2f}")
        st.write(f"- **Price Volatility (Std. Dev.)**: ${row['Price Volatility']:.2f}")
        st.write(f"- **Max Drawdown**: {row['Max Drawdown (%)']:.2f}%")
        st.write("----")  # Add a separator between companies

def data_exploration():
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")
    
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        
        # Check if stock_df is not empty
        if stock_df.empty:
            st.warning(f"No data available for {stock}. Please select a different stock.")
            return
        
        show_stock_trend(stock, stock_df)

def show_stock_trend(stock, stock_df):
    st.write(f"### {stock} Stock Trends")
    
    # Check if 'date' and 'close' columns exist
    if 'date' in stock_df.columns and 'close' in stock_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Close Price', line=dict(color='cyan')))
        fig.update_layout(title=f"{stock} Stock Closing Price", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend note logic
        if stock_df['close'].iloc[-1] > stock_df['close'].iloc[0]:
            trend_note = 'Stock is on a solid upward trend. Investing here might be profitable.'
        else:
            trend_note = 'Stock has been trending downwards. Caution is advised.'
        
        st.markdown(f"Trend Note: {trend_note}")
    else:
        st.error(f"Data for {stock} is missing required columns.")

        

def strategy_simulation():
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    selected_name = st.selectbox("Select Company Name", names)

    if selected_name:
        df = data_prep(data, selected_name)
        
        # Get unique years from the dataset for dynamic selection
        df['date'] = pd.to_datetime(df['date'])
        years = df['date'].dt.year.unique().tolist()
        years.sort()

        # Year selection based on dataset
        selected_year = st.selectbox("Select Year", years)

        # Filter data based on selected year
        df_selected_year = df[df['date'].dt.year == selected_year]

        initial_investment = st.number_input("Enter your initial investment ($)", value=1000, step=100)
        if st.button("Start Simulation"):
            net_worth_history = test_stock(df_selected_year, initial_investment, num_episodes=100)
            plot_net_worth(net_worth_history, df_selected_year)
            metrics = calculate_performance_metrics(net_worth_history, initial_investment)
            display_performance_metrics(metrics)

if __name__ == "__main__":
    main()
