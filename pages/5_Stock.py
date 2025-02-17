import pandas as pd 
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import *
import yfinance as yf
import os
import folium
from numpy.random import seed
import sys
from openpyxl import load_workbook
import comtypes.client
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import time
import requests
from io import BytesIO
from fredapi import Fred
import json
import unicodedata
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder



def stock_tab_layout():
# Display the logo
    st.set_page_config(page_title="Search", layout="wide")

    if st.button("Back to Portoflio"):
        st.switch_page(r"\\192.168.100.252\Shared\15 - Research\000_Clean\Vinicius\Portfolio Analyzer\pages\1_Portfolio.py")


    temp_dir = r"\\192.168.100.252\Shared\15 - Research\000_Clean\Vinicius\Portfolio Analyzer"
    # Read the CSV files
    final_df = pd.read_csv(os.path.join(temp_dir, 'final_df.csv'))
    all_news = pd.read_csv(os.path.join(temp_dir, 'all_news.csv'))
    sector_net_debt_ebitda = final_df.groupby('Sector')['Net Debt / EBITDA'].median()
    sector_cds = final_df.groupby('Sector')['Issuer Default Risk'].median()

    if not "selected_stock" in st.session_state:
        st.warning("No stock selected. Please go back and select a stock from the dashboard.")
    
    else:
        selected_stock = st.session_state["selected_stock"]
        #Chnage equity to Equity 
        selected_stock = selected_stock.replace("equity", "Equity")
        # Extract the selected stock data
        stock_row = final_df[final_df['Ticker'] == selected_stock]
        stock_name = stock_row["Name"].values[0] if not stock_row.empty else None
        st.title(stock_name)

        # Extract Yahoo ticker
        yahoo_ticker = stock_row['Ticker Yahoo'].values[0] if not stock_row.empty and pd.notnull(stock_row['Ticker Yahoo'].values[0]) else None
        if yahoo_ticker:
            historical_data = yf.download(yahoo_ticker, period="1y")

            if not historical_data.empty:
                # Calculate Bollinger Bands
                historical_data["SMA"] = historical_data["Close"].rolling(window=20).mean()
                historical_data["Upper Band (2*std)"] = historical_data["SMA"] + 2 * historical_data["Close"].rolling(window=20).std()
                historical_data["Lower Band (2*std)"] = historical_data["SMA"] - 2 * historical_data["Close"].rolling(window=20).std()
                historical_data["Upper Band (1*std)"] = historical_data["SMA"] + historical_data["Close"].rolling(window=20).std()
                historical_data["Lower Band (1*std)"] = historical_data["SMA"] - historical_data["Close"].rolling(window=20).std()

                # Create candlestick chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=historical_data.index,
                                            open=historical_data['Open'],
                                            high=historical_data['High'],
                                            low=historical_data['Low'],
                                            close=historical_data['Close'],
                                            name='Candlesticks'))

                # Add Bollinger Bands
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Upper Band (2*std)'], 
                                        line=dict(color='green', width=1), name='Upper Band (2*std)'))
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Lower Band (2*std)'], 
                                        line=dict(color='red', width=1), name='Lower Band (2*std)'))

                # Add the 1*std Bollinger Bands
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Upper Band (1*std)'], 
                                        line=dict(color='lightgreen', width=1, dash='dash'), name='Upper Band (1*std)'))
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Lower Band (1*std)'], 
                                        line=dict(color='lightcoral', width=1, dash='dash'), name='Lower Band (1*std)'))

                # Add 20-Day SMA
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['SMA'], 
                                        line=dict(color='orange', width=1), name='20-Day SMA'))

                # Add Volume as a Bar Chart Below
                fig.add_trace(go.Bar(x=historical_data.index, y=historical_data['Volume'], 
                                    marker=dict(color='gray', opacity=0.3), yaxis='y2', name='Volume'))

                # Update layout for volume
                fig.update_layout(yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False))

                # Update layout for the chart with increased width
                fig.update_layout(
                    title=f"{stock_name} ({yahoo_ticker}) - Candlestick Chart",
                    yaxis_title="Stock Price",
                    xaxis_title="Date",
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified",
                    height=500,
                    width=1200,
                    legend=dict(x=0.01, y=0.99)
                )

                # Layout for graph and tactical details
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    last_signal_date, last_signal_type = None, None
                    for i in range(len(historical_data) - 1, -1, -1):
                        row = historical_data.iloc[i]
                        if row["Close"] < row["Lower Band (2*std)"]:
                            last_signal_type = "Buy"
                            last_signal_date = row.name.strftime("%Y-%m-%d")
                            break
                        elif row["Close"] > row["Upper Band (2*std)"]:
                            last_signal_type = "Sell"
                            last_signal_date = row.name.strftime("%Y-%m-%d")
                            break

                    trend = "No Clear Trend"
                    st.markdown(
                        f"**Trend:** {trend}  \n"
                        f"**Last Signal:** {last_signal_type if last_signal_type else 'Neutral'}  \n"
                        f"**Signal Date:** {last_signal_date if last_signal_date else 'N/A'}")
        else:
            st.write("No historical price data available for the selected stock.")

     
    # Add additional graphs in the second row
      # Display additional metrics only if a stock is selected
    if 'stock_row' in locals() and not stock_row.empty:
        st.subheader("Additional Metrics")
    
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Revenue and EPS Graph
            years = [2022, 2023, 2024, 2025, 2026, 2027]
            eps_values = [
                stock_row[f"EPS {year}"].values[0] if not pd.isnull(stock_row[f"EPS {year}"].values[0]) else 0
                for year in years
            ]
            sales_values = [
                stock_row[f"Revenue {year} (Billion USD)"].values[0] if not pd.isnull(stock_row[f"Revenue {year} (Billion USD)"].values[0]) else 0
                for year in years
            ]

            # Create the figure with two y-axes
            fig_revenue_eps = go.Figure()

            # Add Revenue line on the secondary axis
            fig_revenue_eps.add_trace(go.Scatter(
                x=years,
                y=sales_values,
                name="Revenue (Bn USD)",
                line=dict(color="lightcoral", width=2),
                yaxis="y2"
            ))

            # Add EPS bars on the primary axis
            fig_revenue_eps.add_trace(go.Bar(
                x=years,
                y=eps_values,
                name="EPS",
                marker_color="#33404F",
                yaxis="y1"
            ))

            # Update layout to include both y-axes
            fig_revenue_eps.update_layout(
                title=dict(
                    text="Revenue and EPS",
                    x=0.5,
                    font=dict(size=20)
                ),
                xaxis_title="Year",
                yaxis=dict(
                    title="EPS",
                    side="left"
                ),
                yaxis2=dict(
                    title="Revenue (Bn USD)",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                legend=dict(x=0.5, y=-0.2, orientation="h"),
                barmode="overlay",
                hovermode="x unified"
            )

            st.plotly_chart(fig_revenue_eps, use_container_width=True)


        with col2:
            # PE Ratio Graph
            years_pe = [2023, 2024, 2025]
            pe_values = [
                stock_row[f"PE Ratio {year}"].values[0] if not pd.isnull(stock_row[f"PE Ratio {year}"].values[0]) else 0
                for year in years_pe
            ]
            fig_pe = go.Figure()
            fig_pe.add_bar(x=years_pe, y=pe_values, name="PE Ratio", marker_color="#33404F")
            fig_pe.update_layout(title=dict(text="PE Ratio", x=0.5, font=dict(size=20)), xaxis_title="Year", yaxis_title="PE Ratio")
            st.plotly_chart(fig_pe, use_container_width=True)

        with col3:
            # Net Debt/EBITDA Graph
            stock_value = stock_row["Net Debt / EBITDA"].values[0] if not pd.isnull(stock_row["Net Debt / EBITDA"].values[0]) else 0
            sector = stock_row["Sector"].values[0]
            sector_value = sector_net_debt_ebitda.get(sector, None)
            fig_debt = go.Figure()
            fig_debt.add_bar(x=["Company", "Sector"], y=[stock_value, sector_value], marker_color=["#33404F", "#FF6F61"])
            fig_debt.update_layout(title=dict(text="Net Debt / EBITDA", x=0.5, font=dict(size=20)), yaxis_title="Ratio")
            st.plotly_chart(fig_debt, use_container_width=True)

        with col4:
            # CDS Spread Graph
            # CDS Spread Graph
            stock_value = stock_row["Issuer Default Risk"].values[0] if not pd.isnull(stock_row["Issuer Default Risk"].values[0]) else 0
            sector = stock_row["Sector"].values[0]
            sector_value = sector_cds.get(sector, None)
            fig_cds = go.Figure()
            fig_cds.add_bar(x=["Company", "Sector"], y=[stock_value, sector_value], marker_color=["#33404F", "#FF6F61"])
            fig_cds.update_layout(title=dict(text="CDS Spread", x=0.5, font=dict(size=20)), yaxis_title="Basis Points")
            st.plotly_chart(fig_cds, use_container_width=True)

stock_tab_layout()
