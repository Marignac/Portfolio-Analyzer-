import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import fmin
from scipy.stats import norm

end_date = (datetime.today()- timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

def upload_tab():    

    st.header("Upload Portfolio with yahoo Ticker")

    # File uploader for portfolio CSV
    uploaded_file = st.file_uploader("Upload your portfolio CSV file (Ticker col1, Weight col2):", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded portfolio into a DataFrame
        uploaded_portfolio = pd.read_csv(uploaded_file)
        uploaded_portfolio = uploaded_portfolio.dropna(subset=['Ticker']) 
        uploaded_portfolio = uploaded_portfolio.iloc[:, :2]

        # Ensure the uploaded file has the correct format
        required_columns = ['Ticker', 'Weight']
        if all(col in uploaded_portfolio.columns for col in required_columns):
            # Display the uploaded portfolio
            st.write("### Uploaded Portfolio")
            st.dataframe(uploaded_portfolio.set_index(uploaded_portfolio.columns[0]))

            # Ensure weights sum to 100%
            total_uploaded_weight = uploaded_portfolio['Weight'].sum()
            if total_uploaded_weight > 100.0000000001:
                st.error("Total portfolio weight exceeds 100%. Please adjust weights in the uploaded file.")
            else:
                # Generate the pie chart for portfolio allocation
                # Layout for Performance and Visualization
                col1, col2 = st.columns([2, 1])  # Allocate 2/3 of space for performance and 1/3 for statistics

                # Performance Graph
                with col1:
                    # Calculate portfolio performance metrics
                    tickers = uploaded_portfolio['Ticker'].tolist()
                    weights = uploaded_portfolio['Weight'] / 100  # Convert weights to fractions

                    # Download historical data for uploaded portfolio
                    uploaded_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                    uploaded_returns = uploaded_data.pct_change()

                    # Calculate portfolio returns
                    portfolio_weights = weights.values
                    portfolio_returns = (uploaded_returns * portfolio_weights).sum(axis=1)

                    # EuroStoxx performance for comparison
                    eurostoxx_data = yf.download("SX5EEX.VI", start=start_date, end=end_date)['Adj Close']
                    eurostoxx_returns = eurostoxx_data.pct_change()

                    # Plot cumulative performance
                    st.write("### Portfolio Performance")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot((1 + portfolio_returns).cumprod(), label="Uploaded Portfolio", linewidth=1.5)
                    ax.plot((1 + eurostoxx_returns).cumprod(), label="EuroStoxx", linewidth=1.5, linestyle="--")
                    ax.set_title("Uploaded Portfolio Performance vs EuroStoxx")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Cumulative Returns")
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    def calculate_stats(returns, confidence_level=0.95):
                        cumulative_return = (1 + returns).prod() - 1  # As a fraction
                        avg_monthly_return = returns.mean() * 22  # As a fraction
                        avg_monthly_volatility = returns.std() * np.sqrt(22) * 100  # Percentage

                        # Sharpe Ratio
                        sharpe_ratio = (
                            (avg_monthly_return * 12) / ((avg_monthly_volatility / 100) * np.sqrt(12))
                            if avg_monthly_volatility != 0 else np.nan
                            )

                        # Sortino Ratio
                        downside_returns = returns[returns < 0]
                        downside_volatility = downside_returns.std() * np.sqrt(22) * 100
                        sortino_ratio = (
                        (avg_monthly_return * 12) / ((downside_volatility / 100) * np.sqrt(12))
                        if downside_volatility != 0 else np.nan
                        )

                        # Max Drawdown
                        cumulative_returns = (1 + returns).cumprod()
                        rolling_max = cumulative_returns.cummax()
                        drawdown = (cumulative_returns / rolling_max) - 1
                        max_drawdown = drawdown.min() * 100  # Convert to percentage

                        # Non-parametric Value at Risk (VaR) - Monthly
                        sorted_returns = np.sort(returns)
                        var_index = int((1 - confidence_level) * len(sorted_returns))
                        non_parametric_var = sorted_returns[var_index] * np.sqrt(22) * 100  # Scale to monthly and convert to percentage

                        # Non-parametric Expected Shortfall (ES) - Monthly
                        tail_losses = sorted_returns[:var_index + 1]
                        if len(tail_losses) > 0:
                            non_parametric_es = tail_losses.mean() * np.sqrt(22) * 100  # Scale to monthly and convert to percentage
                        else:
                            non_parametric_es = np.nan

                        return {
                            "1-Year Return": cumulative_return,  # Fraction
                            "Avg Monthly Return": avg_monthly_return,  # Fraction
                            "Avg Monthly Volatility (%)": avg_monthly_volatility,  # Percentage
                            "Sharpe Ratio": sharpe_ratio,
                            "Sortino Ratio": sortino_ratio,
                            "Max Drawdown (%)": max_drawdown,
                            "Non-parametric Monthly VaR (%)": non_parametric_var,
                            "Non-parametric Monthly ES (%)": non_parametric_es,
                        }
                    
                    portfolio_stats = calculate_stats(portfolio_returns)
                    eurostoxx_stats = calculate_stats(eurostoxx_returns)

                    # Display Performance Statistics Comparison
                    stats_df = pd.DataFrame({
                        "Metric": portfolio_stats.keys(),
                        "Portfolio": [f"{val:.2%}" if "Return" in key else f"{val:.2f}" for key, val in portfolio_stats.items()],
                        "EuroStoxx": [f"{eurostoxx_stats[key]:.2%}" if "Return" in key else f"{eurostoxx_stats[key]:.2f}"
                                    for key in portfolio_stats.keys()]
                    })

                    st.markdown("### Performance Statistics Comparison")
                    st.markdown(
                        stats_df.to_html(
                            index=False,  # Hide the index
                            justify='center',  # Center align the content
                            border=0,  # No table borders
                            escape=False  # Ensure proper HTML rendering
                        ),
                        unsafe_allow_html=True
                    )

            # Pie Chart for Portfolio Allocation
            fig, ax = plt.subplots(figsize=(12, 12))

            # Generate the pie chart
            labels = uploaded_portfolio['Ticker']
            wedges, texts, autotexts = ax.pie(
                uploaded_portfolio['Weight'],
                labels=labels,
                autopct='%1.1f%%',
                startangle=140,
                pctdistance=0.85  # Position percentages closer to the edge inside the pie
            )

            # Format the text on the chart
            for text in texts:
                text.set_fontsize(12)  # Smaller font for tickers

            for autotext in autotexts:
                autotext.set_fontsize(10)  # Smaller font for percentages
                autotext.set_color('black')  # Ensure percentages are visible

            # Set title and subtitle with number of positions
            num_positions = uploaded_portfolio['Ticker'].nunique()
            ax.set_title(f"Portfolio Allocation - {num_positions} Positions", fontsize=16)

            # Create three columns and display the pie chart in the middle one
            col1, col2, col3 = st.columns([1, 3, 1])  # Adjust column widths as needed
            with col2:
                st.pyplot(fig)  # Display the pie chart in the center column

            # Close the figure to free memory
            plt.close(fig)  


upload_tab()