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
import time


### ======================================= DATA PREPARATION ============================================== ###

# Define dynamic start and end dates
end_date = (datetime.today()- timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

temp_dir = r"\\192.168.100.252\Shared\15 - Research\000_Clean\Vinicius\Portfolio Analyzer"
final_df = pd.read_csv(os.path.join(temp_dir, 'final_df.csv'))

# Load Excel file with large-cap weights
df = pd.read_excel(r'S:/15 - Research/000_Clean/Vinicius/Indexes/STOXX_Holdings.xlsx')
df['Weight (%)'] = df['Weight (%)'].astype(float)
# Define ticker-to-company mapping
ticker_to_company = {
    'AI.PA': 'Air Liquide', 'AD.AS': 'Ahold Delhaize N.V.', 'ADYEN.AS': 'Adyen N.V.', 'AIR.PA': 'Airbus SE', 'ALV.DE': 'Allianz SE',
    'ASML.AS': 'ASML Holding NV', 'ABI.BR': 'Anheuser-Busch InBev', 'ADS.DE': 'adidas AG', 'BNP.PA': 'BNP Paribas', 'BBVA.MC': 'BBVA',
    'BN.PA': 'Danone', 'BAS.DE': 'BASF SE', 'BAYN.DE': 'Bayer AG', 'BMW.DE': 'BMW AG',
    'CS.PA': 'AXA SA', 'DHL.DE': 'DHL Group', 'DB1.DE': 'Deutsche Börse AG', 'DTE.DE': 'Deutsche Telekom AG',
    'DG.PA': 'Vinci', 'EL.PA': 'EssilorLuxottica', 'ENI.MI': 'Eni SpA.', 'ENEL.MI': 'Enel SpA',
    'IBE.MC': 'Iberdrola S.A.', 'IFX.DE': 'Infineon Technologies AG', 'INGA.AS': 'ING Groep N.V.',
    'ISP.MI': 'Intesa Sanpaolo S.p.A.', 'ITX.MC': 'Inditex', 'KER.PA': 'Kering', 'MC.PA': 'LVMH',
    'OR.PA': 'L\'Oréal', 'MUV2.DE': 'Munich Re', 'MBG.DE': 'Mercedes-Benz Group AG', 'NDA-FI.HE': 'Nordea Bank Abp',
    'NOKIA.HE': 'Nokia Oyj', 'PRX.AS': 'Prosus N.V.', 'RACE.MI': 'Ferrari N.V.', 'RI.PA': 'Pernod Ricard',
    'RMS.PA': 'Hermès International', 'SAN.MC': 'Banco Santander, S.A.', 'SAN.PA': 'Sanofi', 'SAP.DE': 'SAP SE',
    'SAF.PA': 'Safran SA', 'SGO.PA': 'Saint-Gobain', 'SIEGY': 'Siemens AG', 'STLAM.MI': 'Stellantis N.V.',
    'SU.PA': 'Schneider Electric SE', 'TTE.PA': 'TotalEnergies SE', 'UCG.MI': 'UniCredit SpA.',
    'VOW.DE': 'Volkswagen AG', 'WKL.AS': 'Wolters Kluwer N.V.'
}


# Additional stock options including EuroStoxx (SX5EEX.VI)
additional_tickers = {
    'FRO': 'Frontline', 'SDZ.SW': 'Sandoz', 'AVOL.SW': 'Avolta', 'ZAL.DE': 'Zalando', 'LISP.SW': 'Lindt',
    'UBS': 'UBS', 'VIE.PA': 'Veolia', 'RIO': 'Rio Tinto', 'KGX.DE': 'Kion', 'VNA.DE': 'Vonovia'
}

#Calculate correlation with EuroStoxx for small-cap stocks for Options 3 and 4
df_smallcap = df[df['Weight (%)'] < 1.64]
smallcap_tickers = df_smallcap['Ticker'].tolist()

# Download historical 

max_retries = 5
retry_count = 0
eurostoxx_data = None

while retry_count < max_retries:
    try:
        eurostoxx_data = yf.download("^STOXX50E", start=start_date, end=end_date)['Adj Close']
        if not eurostoxx_data.empty:
            break  # Exit loop if data is successfully fetched
    except KeyError as e:
        print(f"Retry {retry_count + 1}: {e}")
    time.sleep(2)  # Wait 1 second before retrying
    retry_count += 1

if eurostoxx_data is None or eurostoxx_data.empty:
    raise ValueError(f"Failed to retrieve data for {ticker} after {max_retries} retries")


smallcap_data = yf.download(smallcap_tickers, start=start_date, end=end_date)['Adj Close']
smallcap_data.index = smallcap_data.index.tz_localize(None)
eurostoxx_data.index = eurostoxx_data.index.tz_localize(None)
smallcap_returns = smallcap_data.pct_change()
eurostoxx_returns = eurostoxx_data.pct_change()

# Calculate correlation with EuroStoxx
correlations = smallcap_returns.corrwith(eurostoxx_returns).sort_values(ascending=False)
top_5_correlated_tickers = correlations.head(5).index.tolist()
# Option 3: Equal weight for top 5 correlated tickers
equal_weight_option = [1 / 5] * 5

# Option 4: Correlation-weighted for top 5 correlated tickers
total_correlation = correlations.head(5).sum()
corr_weight_option = [corr / total_correlation for corr in correlations.head(5)]

# Mapping of tickers to sector names
sector_names = {
    '^SP500-20': 'Industrials', '^GSPE': 'Energy', '^SPSIBK': 'Banks',
    '^SP500-35': 'Health Care', '^SP500-15': 'Materials', '^SP500-25': 'Consumer Discretionary',
    '^SP500-30': 'Consumer Staples', 'CL=F': 'Crude Oil', 'GC=F': 'Gold'
}


### ======================================= PAGE LAYOUT ============================================== ###

def build_layout():
    eurostoxx_data = yf.download("^STOXX50E", start=start_date, end=end_date)['Adj Close']
    # Set page title
    st.set_page_config(page_title="Build Portofolio", layout="wide")
    st.title("EGL Portfolio Builder")

    st.header("Part 1 - Core")

    # Display large-cap stocks with weights and allow users to remove selections
    df['Company'] = df['Ticker'].map(ticker_to_company)
    print(df)

    df_filtered = df[['Ticker', 'Company', 'Weight (%)']].dropna()
    print(df_filtered.columns)
    df_filtered_largecap = df_filtered[df_filtered['Weight (%)'] > 1.64]

    # Show weights next to each large-cap company name in the dropdown with 2 decimal places
    large_cap_choices = [f"{row['Company']} ({row['Weight (%)']:.2f}%)" for _, row in df_filtered_largecap.iterrows()]
    removed_stocks = st.multiselect("Select large-cap stocks to remove:", large_cap_choices)

    # Filter out removed stocks
    selected_large_caps = [
        row for row in df_filtered_largecap.iterrows()
        if f"{row[1]['Company']} ({row[1]['Weight (%)']:.2f}%)" not in removed_stocks
    ]
    df_filtered_largecap = pd.DataFrame([x[1] for x in selected_large_caps])

    # Calculate total remaining weight in large cap
    print(df_filtered_largecap.columns)
    total_weight_remaining = df_filtered_largecap['Weight (%)'].sum()
    st.write(f"Total Weight Remaining in Large Cap (before multiplier): {total_weight_remaining:.2f}%")

    # Weight multiplier slider
    weight_multiplier = st.slider("Adjust Large Cap Weight Multiplier", min_value=0.5, max_value=1.5, step=0.05)
    adjusted_largecap_weights = df_filtered_largecap['Weight (%)'] * weight_multiplier
    adjusted_total_weight = adjusted_largecap_weights.sum()
    st.write(f"Total Weight in Large Cap (after multiplier): {adjusted_total_weight:.2f}%")

    # Excluded stocks: large-cap removals + df_smallcap
    excluded_stocks = df_filtered[~df_filtered['Ticker'].isin(df_filtered_largecap['Ticker'])]['Ticker'].tolist()
    excluded_stocks.extend(df_smallcap['Ticker'].tolist())
    excluded_stocks = list(set(excluded_stocks))  # Ensure no duplicates

    # Download historical data for excluded stocks
    basket_data = yf.download(excluded_stocks, start=start_date, end=end_date)['Adj Close']
    basket_data.index = basket_data.index.tz_localize(None)
    basket_returns = basket_data.pct_change()

    # Calculate market-weighted returns for the basket
    basket_weights = df[df['Ticker'].isin(excluded_stocks)].set_index('Ticker')['Weight (%)']
    basket_weights /= basket_weights.sum()  # Normalize weights to 1
    market_weighted_basket_returns = (basket_returns * basket_weights).sum(axis=1)

    # Download historical data for the small-cap stocks
    smallcap_data = yf.download(df_smallcap['Ticker'].tolist(), start=start_date, end=end_date)['Adj Close']
    smallcap_data.index = smallcap_data.index.tz_localize(None)
    smallcap_returns = smallcap_data.pct_change()

    # Calculate correlation with the basket for all small-cap stocks
    correlations_to_basket = smallcap_returns.corrwith(market_weighted_basket_returns).sort_values(ascending=False)

    # Define Option 1: Equal weight for the top 5 correlated stocks
    top_5_basket_correlated_tickers = correlations_to_basket.head(5).index.tolist()
    option_1_weights = [1 / len(top_5_basket_correlated_tickers)] * len(top_5_basket_correlated_tickers)

    # Define Option 2: Correlation-weighted for the top 5 correlated stocks
    total_correlation_to_basket = correlations_to_basket.head(5).sum()
    option_2_weights = [
        corr / total_correlation_to_basket
        for corr in correlations_to_basket.head(5)
    ]

    st.header("Part 2 - Anchor")
    # Display options and their weights
    st.write("### Small Cap Options:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**Excluded Basket EW Correlation**")
        df_option_1 = pd.DataFrame({"Ticker": top_5_basket_correlated_tickers, "Weight": option_1_weights})
        df_option_1.index += 1  # Set index to start from 1
        st.table(df_option_1)

    with col2:
        st.write("**Excluded Basket Highest Correlation**")
        df_option_2 = pd.DataFrame({"Ticker": top_5_basket_correlated_tickers, "Weight": option_2_weights})
        df_option_2.index += 1
        st.table(df_option_2)

    with col3:
        st.write("**Index Equal Weighted Correlation**")
        df_option_3 = pd.DataFrame({"Ticker": top_5_correlated_tickers, "Weight": equal_weight_option})
        df_option_3.index += 1
        st.table(df_option_3)

    with col4:
        st.write("**Index Highest Correlation**")
        df_option_4 = pd.DataFrame({"Ticker": top_5_correlated_tickers, "Weight": corr_weight_option})
        df_option_4.index += 1
        st.table(df_option_4)

    # Portfolio allocation choice
    option = st.selectbox("Choose Small Cap Option to Allocate Remaining Weight", ["Option 1", "Option 2", "Option 3", "Option 4"])
    remaining_weight = 100 - adjusted_total_weight
    allocation_percentage = st.slider(f"Allocate remaining weight ({remaining_weight:.2f}%) to {option}:", 0.0, remaining_weight, step=0.5)

    # Select small cap tickers and assign the chosen proportion of the remaining weight
    if option == "Option 1":
        small_cap_tickers = top_5_basket_correlated_tickers
        small_cap_weights = option_1_weights
    elif option == "Option 2":
        small_cap_tickers = top_5_basket_correlated_tickers
        small_cap_weights = option_2_weights
    elif option == "Option 3":
        small_cap_tickers = top_5_correlated_tickers
        small_cap_weights = equal_weight_option
    else:
        small_cap_tickers = top_5_correlated_tickers
        small_cap_weights = corr_weight_option

    df_smallcap_portfolio = pd.DataFrame({
        'Ticker': small_cap_tickers,
        'Weight (%)': [w * (allocation_percentage / sum(small_cap_weights)) for w in small_cap_weights]
    })

    # Calculate remaining weight for other allocations
    remaining_weight_for_additional = remaining_weight - allocation_percentage
    st.write(f"Remaining Weight for Additional Stocks: {remaining_weight_for_additional:.2f}%")


    # Part 3 - Satellite
    st.header("Part 3 - Satellite")
    
    # Correlation heatmap for additional stocks and EuroStoxx
    additional_stock_data = yf.download(list(additional_tickers.keys()), start=start_date, end=end_date)['Adj Close']
    #merge with EuroStoxx data
    additional_stock_data['EuroStoxx'] = eurostoxx_data
    additional_returns = additional_stock_data.pct_change()
    corr_matrix = additional_returns.corr()
    st.write("Correlation Matrix for Additional Stocks including EuroStoxx")
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        annot_kws={"size": 8},  # Annotation font size
        cbar_kws={"shrink": 0.7},  # Shrink the color bar
        ax=ax
    )
    ax.tick_params(axis='both', labelsize=8)  # Adjust tick label size

    # Create three columns and display the heatmap in the center one
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the width ratios as needed
    with col2:
        st.pyplot(fig)  # Display the heatmap in the middle column

    # Close the figure to free memory
    plt.close(fig)
    

    # Select up to 5 additional stocks
    selected_additional = st.multiselect("Select up to 5 additional stocks:", list(additional_tickers.keys()), max_selections=5)

    # Ensure the allocation input has correct format
    additional_weights = []

    if selected_additional:
        satellite_allocation_percentage = st.slider(
            f"Allocate remaining weight ({remaining_weight_for_additional:.2f}%) to Satellite:", 
            0.0, remaining_weight_for_additional, step=0.5
        )
        
        # Display the total allocation available
        additional_weights_input = st.text_input("Enter weights for selected additional stocks (comma-separated, should sum to 1):")
        try:
            # Process the input weights
            input_weights = [float(w) for w in additional_weights_input.split(",")]
            if len(input_weights) == len(selected_additional) and sum(input_weights) == 1:
                # Scale weights to the slider allocation percentage
                additional_weights = [w * (satellite_allocation_percentage / sum(input_weights)) for w in input_weights]
                valid_input = True
            else:
                st.error("Please ensure weights sum to 1 and match the number of selected stocks.")
                valid_input = False
        except:
            st.error("Please enter valid numeric weights.")
            valid_input = False
    else:
        valid_input = False

    # Recalculate the remaining weight for Part 4 - Wild Card based on Satellite allocation
    remaining_weight_for_wildcard = remaining_weight_for_additional - satellite_allocation_percentage
    st.write(f"Remaining Weight after Satellite Allocation: {remaining_weight_for_wildcard:.2f}%")

    # Additional stocks DataFrame, only created if input is valid
    if valid_input:
        df_additional = pd.DataFrame({'Ticker': selected_additional, 'Weight (%)': additional_weights})
    else:
        df_additional = pd.DataFrame(columns=['Ticker', 'Weight (%)'])


        # Part 4 - Wild Card
    st.header("Part 4 - Wild Card")

    # Calculate the remaining weight after Part 3 to allocate fully in Part 4
    remaining_weight_for_wildcard = remaining_weight_for_additional - satellite_allocation_percentage

    # Prompt the user to enter up to 3 Yahoo tickers for additional customization
    st.write("Enter up to 3 Yahoo tickers for additional customization:")

    # Collect tickers from user input
    wildcard_tickers = []
    cols = st.columns(3)

    # Input ticker names in columns
    for i, col in enumerate(cols):
        ticker = col.text_input(f"Ticker {i + 1}:", key=f"wildcard_ticker_{i}")
        if ticker:
            wildcard_tickers.append(ticker)

    # Adjust weight allocation dynamically based on the number of selected tickers
    if wildcard_tickers:
        num_tickers = len(wildcard_tickers)
        wildcard_weights = [0] * num_tickers  # Initialize weights based on the number of tickers

        # If only one ticker is selected, assign it the full remaining weight
        if num_tickers == 1:
            wildcard_weights[0] = remaining_weight_for_wildcard

        # If two or three tickers are selected, display sliders to allocate the weight
        else:
            remaining = remaining_weight_for_wildcard  # Track remaining weight for distribution
            for i, col in enumerate(cols[:num_tickers]):
                max_value = remaining  # Set max to remaining weight at each step
                # Distribute weight allocation with dynamic max based on remaining weight
                wildcard_weights[i] = col.slider(
                    f"Allocate weight for {wildcard_tickers[i]}:", 
                    min_value=0.0, max_value=max_value, 
                    value=remaining / (num_tickers - i), step=0.1, key=f"wildcard_weight_{i}"
                )
                # Update remaining weight after each slider adjustment
                remaining -= wildcard_weights[i]

    # Combine ticker and weight data into DataFrame
    df_wildcard = pd.DataFrame({
        'Ticker': wildcard_tickers,
        'Weight (%)': wildcard_weights
    })


    # Build DataFrames for each part with final weights
    df_largecap_portfolio = df_filtered_largecap[['Ticker', 'Weight (%)']].copy()
    df_largecap_portfolio['Weight (%)'] = adjusted_largecap_weights

    # Small cap portfolio for Anchor and Satellite
    df_smallcap_portfolio = pd.DataFrame({
        'Ticker': small_cap_tickers,
        'Weight (%)': [w * (allocation_percentage / sum(small_cap_weights)) for w in small_cap_weights]
    })

    # Additional stocks
    df_additional = pd.DataFrame({'Ticker': selected_additional, 'Weight (%)': additional_weights})

    # Concatenate all parts into one final portfolio
    portfolio_df = pd.concat([df_largecap_portfolio, df_smallcap_portfolio, df_additional, df_wildcard])

    # Ensure total weight does not exceed 100%
    total_allocated_weight = portfolio_df['Weight (%)'].sum()
    if total_allocated_weight > 100:
        st.error("Total portfolio weight exceeds 100%. Please adjust weights.")
    else:
        
        # Layout for Performance and Visualization
        col1, col2 = st.columns([2, 1])  # Allocate 2/3 of space for performance and 1/3 for statistics

        # Performance Graph
        with col1:
            # Retrieve historical data and calculate performance
            tickers = portfolio_df['Ticker'].tolist()
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            returns = data.pct_change()

            # Portfolio returns calculation
            portfolio_weights = portfolio_df.set_index('Ticker')['Weight (%)'] / 100
            portfolio_returns = (returns * portfolio_weights).sum(axis=1)

            # EuroStoxx performance for comparison
            
            eurostoxx_returns = eurostoxx_data.pct_change()

            # Plot cumulative performance
            st.write("### Portfolio Performance")
            fig, ax = plt.subplots(figsize=(12, 5))  # Adjust height for better display
            ax.plot((1 + portfolio_returns).cumprod(), label="Portfolio", linewidth=1.5)
            ax.plot((1 + eurostoxx_returns).cumprod(), label="EuroStoxx", linewidth=1.5, linestyle="--")
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
        labels = portfolio_df['Ticker']
        wedges, texts, autotexts = ax.pie(
            portfolio_df['Weight (%)'],
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
        num_positions = portfolio_df['Ticker'].nunique()
        ax.set_title(f"Portfolio Allocation - {num_positions} Positions", fontsize=16)

        # Create three columns and display the pie chart in the middle one
        col1, col2, col3 = st.columns([1, 3, 1])  # Adjust column widths as needed
        with col2:
            st.pyplot(fig)  # Display the pie chart in the center column

        # Close the figure to free memory
        plt.close(fig)

    with col3:
        # Prepare data for export
        portfolio_export = portfolio_df.copy()
        portfolio_export = portfolio_export.rename(columns={'Ticker': 'Ticker', 'Weight (%)': 'Weight'})
        portfolio_export['Weight'] = portfolio_export['Weight'].round(2)

        # Calculate additional metrics for export
        portfolio_export['Performance Metrics'] = portfolio_stats["1-Year Return"]

        # Combine performance metrics into the export DataFrame
        portfolio_export['1-Year Return'] = f"{portfolio_stats['1-Year Return']:.2%}"
        portfolio_export['Sharpe Ratio'] = f"{portfolio_stats['Sharpe Ratio']:.2f}"

        # Convert DataFrame to CSV
        csv_data = portfolio_export.to_csv(index=False)

        # Export button
        st.download_button(
            label="Export Portfolio to CSV",
            data=csv_data,
            file_name="portfolio_export.csv",
            mime="text/csv"
        )

build_layout()