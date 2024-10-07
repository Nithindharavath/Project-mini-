import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Cache the data preparation function for optimization
@st.cache_data
def data_prep(data, name):
    df = data[data['Name'] == name].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    df['5day_MA'][:4] = 0  # Initialize first few values as 0 for simplicity
    return df

# Function to display stock company overview
def company_overview():
    st.write("### Stock Company Overview")

    # Load stock data
    try:
        data = pd.read_csv('all_stocks_5yr.csv')
        st.write("Data loaded successfully!")
    except Exception as e:
        st.write(f"Error loading data: {e}")
        return

    stock = st.sidebar.selectbox("Choose Company Stocks", list(data['Name'].unique()), index=0)

    if stock:
        try:
            stock_df = data_prep(data, stock)
            st.write("Stock data prepared!")
        except Exception as e:
            st.write(f"Error processing stock data: {e}")
            return

        # Display company profile (Dummy data for demonstration)
        st.subheader(f"{stock} Overview")
        st.write(f"**Sector**: Technology")  # Example sector
        st.write(f"**Industry**: Software & IT Services")  # Example industry
        st.write(f"**Market Cap**: $500 Billion")  # Example market cap
        st.write(f"**P/E Ratio**: 30.5")  # Example P/E ratio
        st.write(f"**Dividend Yield**: 1.5%")  # Example dividend yield

        # Display stock performance details
        st.subheader("Stock Performance Overview")
        st.write(f"**52-Week High**: ${stock_df['close'].max():.2f}")
        st.write(f"**52-Week Low**: ${stock_df['close'].min():.2f}")
        st.write(f"**Current Price**: ${stock_df['close'].iloc[-1]:.2f}")

        # Plot stock trend
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Stock Price', line=dict(color='cyan', width=2)))
            fig.update_layout(title=f'Stock Price Trend for {stock}', xaxis_title='Date', yaxis_title='Price ($)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Error plotting stock data: {e}")

# Function to handle home page content
def home_page():
    st.write("Welcome to the Stock Trading Strategy Application!")
    st.write("This app helps you explore stock trends, simulate trading strategies using reinforcement learning, and view company stock overviews.")

# Function to handle data exploration
def data_exploration():
    st.write("Data Exploration Page: Analyze various stock data.")

# Function to handle strategy simulation (placeholder for future implementation)
def strategy_simulation():
    st.write("Strategy Simulation Page: Enhancing stock trading using reinforcement learning.")

# Main function for the Streamlit app
def main():
    st.title("Enhancing Stock Trading Strategy using Reinforcement Learning")
    
    # Define the tabs available in the sidebar
    tabs = ["Home", "Data Exploration", "Strategy Simulation", "Company Overview"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home":
        home_page()
    elif selected_tab == "Data Exploration":
        data_exploration()
    elif selected_tab == "Strategy Simulation":
        strategy_simulation()
    elif selected_tab == "Company Overview":
        company_overview()

# Execute the main function
if __name__ == '__main__':
    main()
