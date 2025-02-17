import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import fmin
from scipy.stats import norm
import os
import io 
from st_aggrid import AgGrid, JsCode, GridUpdateMode, AgGridTheme, GridOptionsBuilder
from streamlit_modal import Modal


# read excel file called Dict_Yahoo_Bloomberg.xlsx
dict = pd.read_excel(r'S:/15 - Research/000_Clean/Vinicius/Portfolio Analyzer/Dict_Yahoo_Bloom.xlsx')

temp_dir = r"\\192.168.100.252\Shared\15 - Research\000_Clean\Vinicius\Dashboard project\RESEARCH DASH\temp"

# Read the CSV files
final_df = pd.read_csv(os.path.join(temp_dir, 'final_df.csv'))
all_news = pd.read_csv(os.path.join(temp_dir, 'all_news.csv'))


# Define dynamic start and end dates
today = datetime.today()
end_date = (datetime.today()- timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')


def portfolio_layout():
    st.set_page_config(page_title="Portfolio", layout="wide")
    st.title("Portfolio Analysis")
    st.header("Upload Decaf File")
     
    if 'uploaded_portfolio_file' not in st.session_state:
        st.session_state['uploaded_portfolio_file'] = None

    # File uploader for portfolio CSV
    uploaded_file = st.file_uploader("Upload your DECAF portfolio CSV file:", type=["csv"])

    if uploaded_file is not None:
        # Save the uploaded file in session state
        st.session_state['uploaded_portfolio_file'] = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded and saved in session.")

    # Check if a file exists in session state
    if st.session_state['uploaded_portfolio_file'] is not None:
        try:
            file = st.session_state['uploaded_portfolio_file']
            portfolio = file.copy()
            portfolio = portfolio[(portfolio['Classification 1'] == 'Equities') | (portfolio['Classification 1'] == 'Commodities')]
            portfolio = portfolio[['Name', 'Portfolio', 'Symbol', 'ISIN', 'Value']]
            portfolio.rename(columns={'Portfolio': 'Strategy'}, inplace=True)
            total_value = portfolio['Value'].sum()
            portfolio['Weight'] = portfolio['Value'] / total_value
            uploaded_portfolio = portfolio.dropna(subset=['Weight'])
            uploaded_portfolio['Symbol'] = uploaded_portfolio['Symbol'].str.replace(" GY ", " GR ")
            uploaded_portfolio = uploaded_portfolio[~uploaded_portfolio['Symbol'].str.contains('Index')]

            # Create a mapping from the dictionary
            ticker_mapping = dict.set_index('Bloomberg')['Yahoo'].to_dict()

            # Map the 'Symbol' column in uploaded_portfolio to the 'Yahoo' column in dict
            uploaded_portfolio['Ticker'] = uploaded_portfolio['Symbol'].map(ticker_mapping)

            # Handle unmapped symbols
            if uploaded_portfolio['Ticker'].isnull().any():
                st.warning("Some symbols in the uploaded portfolio could not be mapped to Yahoo tickers.")

            uploaded_portfolio['Bloom'] = uploaded_portfolio['Symbol'].str.replace('Equity', '', regex=True).str.strip()

            # Determine the benchmark based on strategy
            if "European" in uploaded_portfolio['Strategy'].iloc[0]:
                benchmark_ticker = "SX5EEX.VI"  # EuroStoxx
                benchmark_name = "EuroStoxx"
            elif "World" in uploaded_portfolio['Strategy'].iloc[0]:
                benchmark_ticker = "URTH"  # MSCI World
                benchmark_name = "MSCI World"
            else:
                benchmark_ticker = ""
                benchmark_name = "Benchmark"

            # Ensure the uploaded file has the correct format
            required_columns = ['Ticker', 'Weight']
            if all(col in uploaded_portfolio.columns for col in required_columns):
                # Display the uploaded portfolio
            
                # Ensure weights sum to 100%
                total_uploaded_weight = uploaded_portfolio['Weight'].sum()
                if total_uploaded_weight > 1.1000000001:
                    st.error("Total portfolio weight exceeds 100%. Please adjust weights in the uploaded file.")
                else:
                                    # Define layout
                    col1, col2 = st.columns([5, 3])  # Performance graph (2/3) and table (1/3)

                    # Display performance graph in col1
                    with col1:

                        # Download historical data for uploaded portfolio
                        tickers = uploaded_portfolio['Ticker'].tolist()
                        weights = uploaded_portfolio['Weight']
                        uploaded_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                        uploaded_returns = uploaded_data.pct_change()

                        # Calculate portfolio returns
                        portfolio_weights = weights.values
                        portfolio_returns = (uploaded_returns * portfolio_weights).sum(axis=1)

                        # Benchmark performance for comparison
                        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
                        benchmark_returns = benchmark_data.pct_change()

                        # Plot cumulative performance
                        st.write("### Portfolio Performance")
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot((1 + portfolio_returns).cumprod(), label="Uploaded Portfolio", linewidth=1.5)
                        ax.plot((1 + benchmark_returns).cumprod(), label=benchmark_name, linewidth=1.5, linestyle="--")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Cumulative Returns")
                        ax.legend()
                        st.pyplot(fig)
                        
                    with col2:
                        # Performance statistics calculation
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

                        uploaded_stats = calculate_stats(portfolio_returns)
                        benchmark_stats = calculate_stats(benchmark_returns)

                        # Display performance statistics comparison
                        stats_df = pd.DataFrame({
                            "Metric": uploaded_stats.keys(),
                            "Uploaded Portfolio": [f"{val:.2%}" if "Return" in key else f"{val:.2f}" for key, val in uploaded_stats.items()],
                            benchmark_name: [f"{benchmark_stats[key]:.2%}" if "Return" in key else f"{benchmark_stats[key]:.2f}" for key in uploaded_stats.keys()]
                        })

                        # Display the table as clean HTML
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
                    # Place the pie chart at the bottom
                    
                    fig, ax = plt.subplots(figsize=(12, 12))

                    # Generate the pie chart
                    labels = uploaded_portfolio['Bloom']
                    wedges, texts, autotexts = ax.pie(
                        uploaded_portfolio['Weight'],
                        labels=labels,
                        autopct='%1.1f%%',
                        startangle=140,
                        pctdistance=0.9  # Position percentages closer to the edge inside the pie
                    )

                    # Format the text on the chart
                    for text in texts:
                        text.set_fontsize(12)  # Smaller font for tickers

                    for autotext in autotexts:
                        autotext.set_fontsize(10)  # Smaller font for percentages
                        autotext.set_color('black')  # Ensure percentages are visible

                    # Set title
                    ax.set_title("Uploaded Portfolio Allocation", fontsize=16)

                    # Create three columns and display the pie chart in the middle one
                    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths as needed
                    with col2:
                        st.pyplot(fig)  # Display the pie chart in the center column

                    # Close the figure to free memory
                    plt.close(fig)
                    merged_df = uploaded_portfolio.merge(final_df, on='ISIN', how='inner')
                    columns_to_display = {
                        'Name_x': 'Name',
                        'Symbol': 'Symbol',
                        'Sentiment': 'Cube Score',
                        'ISIN': 'ISIN',
                        'Weight': 'Weight',
                        'Value': 'Value',
                        'Sector': 'Sector',
                        'Country': 'Country',
                        'Market Cap (Billion USD)': 'Market Cap',
                        'FWD PE Ratio': 'FWD PE Ratio',
                        'Dividend Yield': 'Dividend Yield',
                        'P/B Ratio': 'P/B Ratio',
                        'ESG Score': 'ESG Score',
                        'Net Debt / EBITDA': 'Net Debt / EBITDA',
                        'Median Sector Net Debt / EBITDA': 'Median Sector Net Debt / EBITDA',
                        'Issuer Default Risk': 'Issuer Default Risk',
                        'Median Sector Default Risk': 'Median Sector Default Risk',
                        'ROA': 'ROA'
                    }

                    # Filter and rename the columns
                    df_to_display = merged_df[columns_to_display.keys()].rename(columns=columns_to_display)


                    st.markdown("### Constituents")
                    # Configure AgGrid options
                    grid_options = GridOptionsBuilder.from_dataframe(df_to_display)
                    grid_options.configure_selection(selection_mode='single', use_checkbox=True)
                    grid_options.configure_default_column(resizable=False, sortable=False, filterable=False, editable=False)
                    grid_options.configure_grid_options(alwaysShowHorizontalScroll = True)
                    grid_options.configure_grid_options(domLayout='normal')
                    grid_options.configure_grid_options(rowHeight=40)
                    grid_options.configure_column("Name", cellStyle=JsCode('''
                        function(params) {
                            const sentiment = params.data['Cube Score'];
                            if (sentiment === 'Positive') {
                                return { 'color': 'white', 'backgroundColor': '#007000', 'fontSize': '12px', 'padding': '4px' };
                            } else if (sentiment === 'Negative') {
                                return { 'color': 'white', 'backgroundColor': '#D2222D', 'fontSize': '12px', 'padding': '4px' };
                            } else if (sentiment === 'Neutral') {
                                return { 'color': 'black', 'backgroundColor': '#FFD700', 'fontSize': '12px', 'padding': '4px' };
                            } else {
                                return { 'color': 'black', 'backgroundColor': 'white', 'fontSize': '12px', 'padding': '4px' };
                            }
                        }
                    '''))
                    grid_options.configure_column("Symbol", cellStyle=JsCode('''
                        function(params) {
                            const sentiment = params.data['Cube Score'];
                            if (sentiment === 'Positive') {
                                return { 'color': 'white', 'backgroundColor': '#007000', 'fontSize': '12px', 'padding': '4px' };
                            } else if (sentiment === 'Negative') {
                                return { 'color': 'white', 'backgroundColor': '#D2222D', 'fontSize': '12px', 'padding': '4px' };
                            } else if (sentiment === 'Neutral') {
                                return { 'color': 'black', 'backgroundColor': '#FFD700', 'fontSize': '12px', 'padding': '4px' };
                            } else {
                                return { 'color': 'black', 'backgroundColor': 'white', 'fontSize': '12px', 'padding': '4px' };
                            }
                        }
                    '''))
                    grid_options.configure_column("Cube Score", cellStyle=JsCode('''
                        function(params) {
                            const sentiment = params.data['Cube Score'];
                            if (sentiment === 'Positive') {
                                return { 'color': 'white', 'backgroundColor': '#007000', 'fontSize': '12px', 'padding': '4px' };
                            } else if (sentiment === 'Negative') {
                                return { 'color': 'white', 'backgroundColor': '#D2222D', 'fontSize': '12px', 'padding': '4px' };
                            } else if (sentiment === 'Neutral') {
                                return { 'color': 'black', 'backgroundColor': '#FFD700', 'fontSize': '12px', 'padding': '4px' };
                            } else {
                                return { 'color': 'black', 'backgroundColor': 'white', 'fontSize': '12px', 'padding': '4px' };
                            }
                        }
                    '''))

                    # Display the AgGrid table
                    grid_response = AgGrid(
                        df_to_display,
                        gridOptions=grid_options.build(),
                        theme="material",
                        height=400,
                        allow_unsafe_jscode=True,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                    )
                    # Get selected rows
                    selection = grid_response['selected_rows']
                    selected_df = pd.DataFrame(selection)

                    # Store the selected stock in session state
                    if not selected_df.empty:  # Check if a row is selected
                        stock = selected_df["Symbol"].values[0]
                        st.session_state["selected_stock"] = stock
                        if st.button("Go to Stock Page"):
                            # Navigate to the Sector page
                            st.switch_page(r"\\192.168.100.252\Shared\15 - Research\000_Clean\Vinicius\Portfolio Analyzer\pages\5_Stock.py")
                    else:
                        st.warning("Select a checkbox to anaylze a stock")
        except Exception as e:
            st.error(f"An error occurred: {e}")

portfolio_layout()