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

# Get the current working directory

script_dir = os.path.dirname(os.path.realpath(__file__))

os.chdir(script_dir)

# =========================================================================================
# =================================== DATA MANIPULATION ===================================
# =========================================================================================


# ====================================== NEWS =============================================

# Define the network path and file
network_path = r"\\192.168.100.252\Shared\15 - Research\000_Clean\Scoring\News"
file_name = "3_yahoo.xlsm"
file_path = os.path.join(network_path, file_name)

# Load the macro-enabled Excel workbook
workbook = load_workbook(file_path, data_only=True)

# Read the 'Cube' sheet into a pandas DataFrame
sheet_name = "Cube"
sheet = workbook[sheet_name]
data = list(sheet.values)

# Use the second row (index 1) as the header
header_row = 1  # Adjust index for header row
df = pd.DataFrame(data[header_row + 1:], columns=data[header_row])

# Drop rows where all values are None (to remove trailing empty rows)
df.dropna(how="all", inplace=True)

# Strip and clean column names to ensure proper matching
df.columns = df.columns.str.strip()

# Ensure relevant columns are present
relevant_columns = [
    'Ticker Bloom',
    'ISINS',
    'Ticker Yahoo',
    'All News in priority from scrapper to Yahoo',
]
missing_columns = [col for col in relevant_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

df = df[relevant_columns]

# Function to split the news by dates
def split_news(news_entry):
    if pd.isna(news_entry):
        return []
    return [entry.strip() for entry in news_entry.split("\n") if entry]

# Split the 'All News' column into separate entries
df['All News Split'] = df['All News in priority from scrapper to Yahoo'].apply(split_news)

# Create new columns for split news
def create_news_columns(news_split):
    max_splits = max(len(news) for news in news_split if news)
    split_columns = {}
    for i in range(max_splits):
        split_columns[f"News_{i+1}"] = [news[i] if i < len(news) else None for news in news_split]
    return split_columns

news_split_data = create_news_columns(df['All News Split'])
news_split_df = pd.DataFrame(news_split_data)

# Function to split the news entry into Date and Content
def split_news_entry(entry):
    if pd.isna(entry):
        return None, None
    # Split by the first occurrence of ': '
    parts = entry.split(": ", 1)
    if len(parts) == 2:
        date, content = parts
        return date.strip(), content.strip()
    return None, entry.strip()  # Handle cases with no date

# Split each news column into date and content
for col in news_split_df.columns:
    news_split_df[[f"{col}_Date", f"{col}_Content"]] = news_split_df[col].apply(
        lambda x: pd.Series(split_news_entry(x))
    )
    news_split_df.drop(columns=[col], inplace=True)  # Remove the original column

# Concatenate the new columns with the original dataframe
news = pd.concat([df.drop(columns=['All News Split']), news_split_df], axis=1)

# ====================================== ALL NEWS =============================================
def extract_all_news(news_df):
    """Extract ISIN, Date, and Content from news data."""
    # Identify news-related columns
    news_columns = [col for col in news_df.columns if "News_" in col and "Content" in col]
    date_columns = [col for col in news_df.columns if "News_" in col and "Date" in col]

    # Create a list to hold all news entries
    all_news_entries = []

    # Iterate through each row and extract ISIN, Date, and Content
    for _, row in news_df.iterrows():
        for date_col, content_col in zip(date_columns, news_columns):
            if pd.notnull(row[date_col]) and pd.notnull(row[content_col]):
                all_news_entries.append({
                    "ISIN": row.get("ISINS"),
                    "Date": row[date_col].strip().replace('.', '-'),  # Swap '.' for '-'
                    "Content": row[content_col]
                })

    # Convert to DataFrame
    all_news_df = pd.DataFrame(all_news_entries)

    # Drop rows with missing dates
    all_news_df = all_news_df.dropna(subset=["Date"])

    # Sort the DataFrame by date in descending order (assuming format 'dd-mm-yyyy')
    all_news_df = all_news_df.sort_values(by="Date", key=lambda col: pd.to_datetime(col, format='%d-%m-%Y'), ascending=False).reset_index(drop=True)

    return all_news_df

# Extract all news into the new DataFrame
all_news = extract_all_news(news)

def clean_text(text):
    """Cleans text by normalizing and removing unwanted characters."""
    if isinstance(text, str):
        # Normalize unicode characters to remove accents and other artifacts
        text = unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('utf-8')
        # Encode and decode to ensure valid UTF-8
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        # Remove occurrences of x000D
        text = text.replace("_x000D_", "")
        return text.strip()  # Remove leading/trailing whitespaces
    return text

# Apply the updated cleaning function
all_news = all_news.applymap(clean_text)
# =================================== SCORE =============================================

# Define the file path
file_path = r"\\192.168.100.252\Shared\15 - Research\Hugo B\CUBE_Scores_Hugo.xlsm"

try:
    # Read the Excel file with header starting at row 2 (Python index 1)
    data = pd.read_excel(file_path, header=2)
    
    # Select the required columns
    score = data[['ISIN', 'Ticker', 'Score1', 'Score2', 'Score1Sector', 'Score2Sector', 'Score3' , 'Score3Sector', 'Final Score']]
    
    # Drop rows with missing values in the ISIN column
    score = score.dropna(subset=['ISIN'])
    
    # Rename the 'Ticker' column to 'Ticker_No_E'
    score = score.rename(columns={'Ticker': 'Ticker_No_E'})
    
    
except FileNotFoundError:
    print("The file was not found. Please check the path and ensure it is correct.")
except Exception as e:
    print(f"An error occurred: {e}")

# =================================== SQL =============================================

def charger_variables_env():
    """Charge les variables d'environnement depuis un fichier .env."""
    load_dotenv()
    serveur = os.getenv('SERVEUR')
    base_de_donnees = os.getenv('BASE_DE_DONNEES')
    utilisateur = os.getenv('UTILISATEUR')
    mot_de_passe = os.getenv('MOT_DE_PASSE')
    return serveur, base_de_donnees, utilisateur, mot_de_passe

def creer_connexion_sql(serveur, base_de_donnees, utilisateur, mot_de_passe):
    """Crée et retourne une connexion SQLAlchemy à la base de données."""
    try:
        connexion_chaine = f"mssql+pyodbc://{utilisateur}:{mot_de_passe}@{serveur}/{base_de_donnees}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(connexion_chaine)
        print("Connexion SQL établie.")
        return engine
    except SQLAlchemyError as e:
        print("Erreur lors de la création de la connexion SQL :", e)
        return None

def Main():
        # Chargement des variables d'environnement

    serveur='192.168.100.233,1433'
    base_de_donnees='deal-catcher'
    utilisateur='deal-catcher'
    mot_de_passe='deal$2020'

    # Création de la connexion SQL
    engine = creer_connexion_sql(serveur, base_de_donnees, utilisateur, mot_de_passe)
    if engine is None:
        return
        
    df=pd.read_sql('SELECT * FROM CUBE_Data', engine)
    return df

def __init__():
    """Exécute la fonction Main et sauvegarde le DataFrame en CSV."""
    df = Main()  # Appel de la fonction Main pour charger les données
    if df is not None:
        # Sauvegarde du DataFrame en CSV
        csv_file_path = r"S:\15 - Research\000_Clean\Vinicius\Dashboard Project\DASHBOARD\temp\data.csv"  # Chemin du fichier CSV
        df.to_csv(csv_file_path, index=False)
        print(f"Les données ont été sauvegardées dans le fichier {csv_file_path}.")
    else:
        print("Aucune donnée à sauvegarder.")

# Exécution du script
__init__()

df = pd.read_csv(r"S:\15 - Research\000_Clean\Vinicius\Dashboard Project\DASHBOARD\temp\data.csv")

# Function to convert strings with "M", "B", "T", "k", or "%" into numbers
def convert_to_numeric(value):
    try:
        if isinstance(value, str):
            # Remove the '%' sign if present
            value = value.replace('%', '')
            
            # Handle cases for 'M', 'B', 'T', 'k'
            if 'M' in value:
                return float(value.replace('M', '')) * 1e6
            elif 'B' in value:
                return float(value.replace('B', '')) * 1e9
            elif 'T' in value:
                return float(value.replace('T', '')) * 1e12
            elif 'k' in value:
                return float(value.replace('k', '')) * 1e3
            elif '%' in value:
                return float(value.replace('%', '')) 
            else:
                return float(value)  # If no letters, just convert to float
        return value  # If it's already a number, return it as is
    except ValueError:
        return np.nan  # If conversion fails, return NaN
    
# Helper function to convert Excel column letters to zero-based indices
def column_letter_to_index(letter):
    letter = letter.upper()
    index = 0
    for char in letter:
        index = index * 26 + (ord(char) - ord('A') + 1)
    return index - 1

# Define the Excel columns to convert
excel_columns = ['E-S', 'U', 'Z', 'AE', 'AG-CA', 'CC-CY', 'DC', 'DN-DY']

# Dynamically calculate column indices for the ranges
columns_to_convert_indices = []
for col in excel_columns:
    if '-' in col:  # Handle ranges like AP-BP
        start, end = col.split('-')
        columns_to_convert_indices.extend(range(column_letter_to_index(start), column_letter_to_index(end) + 1))
    else:
        columns_to_convert_indices.append(column_letter_to_index(col))

# Apply the conversion function to the specified columns using iloc
for idx in columns_to_convert_indices:
    df.iloc[:, idx] = df.iloc[:, idx].apply(convert_to_numeric)

# Convert necessary columns to numeric once
df[['net_debt_2024', 'Ebitda_2025']] = df[['net_debt_2025', 'Ebitda_2025']].apply(pd.to_numeric, errors='coerce')

# Calculate Net Debt / EBITDA ratio, handling division by zero and NaN values
df['Net Debt / EBITDA'] = df.apply(
    lambda row: row['net_debt_2024'] / row['Ebitda_2025'] if row['Ebitda_2025'] and pd.notnull(row['net_debt_2025']) else None,
    axis=1
)

# Pre-calculate the median Net Debt / EBITDA for each sector
sector_net_debt_ebitda = df.groupby('gics_ind_name')['Net Debt / EBITDA'].median()

# Convert necessary columns to numeric once
df['issuer_default_risk'] = df['issuer_default_risk'].apply(pd.to_numeric, errors='coerce')

# Pre-calculate the median Net Debt / EBITDA for each sector
sector_cds = df.groupby('gics_ind_name')['issuer_default_risk'].median()

### ----------------------------------------------------------------------------------------------------------------------------------------------
stocks = df

# Get today's date
today_date = datetime.today().strftime("%d %b %Y")

def process_data(df):
    """Process data to extract necessary variables and create a new DataFrame."""
    # Convert columns to numeric where necessary
    df['net_debt_2025'] = pd.to_numeric(df['net_debt_2025'], errors='coerce')
    df['Ebitda_2025'] = pd.to_numeric(df['Ebitda_2025'], errors='coerce')
    df['issuer_default_risk'] = pd.to_numeric(df['issuer_default_risk'], errors='coerce')
    
    # Calculate Net Debt / EBITDA
    df['Net Debt / EBITDA'] = df.apply(
        lambda row: row['net_debt_2025'] / row['Ebitda_2025'] if row['Ebitda_2025'] and pd.notnull(row['net_debt_2025']) else None,
        axis=1
    )
    
    # Pre-calculate median Net Debt / EBITDA and default risk by sector
    sector_net_debt_ebitda = df.groupby('gics_ind_name')['Net Debt / EBITDA'].median()
    sector_cds = df.groupby('gics_ind_name')['issuer_default_risk'].median()

    # Extract PE Ratios, Revenue, and EPS as separate columns
    years = [2022, 2023, 2024, 2025, 2026,2027]
    for year in years:
        if f'pe_ratio_{year}' in df.columns:
            df[f'PE Ratio {year}'] = pd.to_numeric(df[f'pe_ratio_{year}'], errors='coerce')
        if f"est_comp_sales_{year}" in df.columns or f"est_sales_{year}" in df.columns:
            df[f"Revenue {year} (Billion USD)"] = df.apply(
                lambda row: round(row.get(f"est_comp_sales_{year}", row.get(f"est_sales_{year}", 0)) / 1_000_000_000, 1) 
                if pd.notnull(row.get(f"est_comp_sales_{year}", row.get(f"est_sales_{year}", None))) 
                else None, 
                axis=1
            )
        if f"eps_{year}" in df.columns or f"est_eps_{year}" in df.columns:
            df[f"EPS {year}"] = df.apply(
                lambda row: row.get(f"eps_{year}", row.get(f"est_eps_{year}", None)),
                axis=1
            )

    # Create the stocks DataFrame
    stocks = pd.DataFrame({
        'Name': df['name'].str.title() if 'name' in df else "N/A",
        'Sector': df['gics_ind_name'],
        'Description': df['Description'] if 'Description' in df else "N/A",
        'Country': df['country_territory_name'],
        'Market Cap (Billion USD)': df['market_cap'].apply(
            lambda x: round(float(x) / 1_000_000_000, 1) if pd.notnull(x) else "N/A"
        ),
        'FWD PE Ratio': df['est_pe_ratio_2026'],
        'Dividend Yield': df['dividend_yield'],
        'ISIN': df['isin'],
        'Ticker': df['ticker'].apply(lambda x: f"{x} Equity" if pd.notnull(x) else "N/A"),
        'P/B Ratio': df['price_to_book'],
        'ESG Score': df['esg_score'],
        'Net Debt / EBITDA': df['Net Debt / EBITDA'],  # Add the calculated column
        'Median Sector Net Debt / EBITDA': df['gics_ind_name'].map(sector_net_debt_ebitda),  # Map sector median
        'Issuer Default Risk': df['issuer_default_risk'],
        'Median Sector Default Risk': df['gics_ind_name'].map(sector_cds),  # Map sector median default risk
        'ROA': df['roa'],
        'Net Debt 2025': df['net_debt_2025'],
        'Net Debt 2024': df['net_debt_2024'],
        'Shares Out 2025': df['shares_out_2025'],
        'Shares Out 2024': df['shares_out_2024'],
        'Current ratio 2025': df['curr_ratio_2025'],
        'Current ratio 2024': df['curr_ratio_2024'],
        'Asset Turnover 2025': df['asset_turnover_2025'],
        'Asset Turnover 2024': df['asset_turnover_2024']
    })

    # Add the PE Ratios, Revenue, and EPS columns for each year to the stocks DataFrame
    for year in years:
        if f'PE Ratio {year}' in df.columns:
            stocks[f'PE Ratio {year}'] = df[f'PE Ratio {year}']
        if f"Revenue {year} (Billion USD)" in df.columns:
            stocks[f"Revenue {year} (Billion USD)"] = df[f"Revenue {year} (Billion USD)"]
        if f"EPS {year}" in df.columns:
            stocks[f"EPS {year}"] = df[f"EPS {year}"]

    return stocks

# Process the data
stocks = process_data(df)

# Merge score and news with stocks by ISIN
try:
    # Ensure the ISIN column exists in all DataFrames
    if 'ISIN' in stocks.columns and 'ISIN' in score.columns and 'ISINS' in news.columns:
        # First merge stocks with score
        merged_data = pd.merge(stocks, score, on='ISIN', how='left')
        
        # Then merge the result with news
        merged_data = pd.merge(merged_data, news, left_on='ISIN', right_on='ISINS', how='left')
        
    else:
        print("One or more DataFrames are missing the required 'ISIN' column.")
except Exception as e:
    print(f"An error occurred while merging: {e}")

merged_data['Sentiment'] =  merged_data['Final Score'].map({
                    3: 'Positive',
                    2: 'Neutral',
                    1: 'Negative'
                })


final_df = merged_data

# read excel file called Dict_Yahoo_Bloomberg.xlsx
dict = pd.read_excel(r'S:/15 - Research/000_Clean/Vinicius/Portfolio Analyzer/Dict_Yahoo_Bloom.xlsx')

# Step 1: Create a mapping from `dict` for Bloomberg to Yahoo
ticker_mapping = dict.set_index('Bloomberg')['Yahoo'].to_dict()

# Step 2: Check for NaN values in 'Ticker Yahoo' in final_df and update using the mapping
final_df['Ticker Yahoo'] = final_df.apply(
    lambda row: ticker_mapping[row['Ticker']] if pd.isna(row['Ticker Yahoo']) and row['Ticker'] in ticker_mapping else row['Ticker Yahoo'], 
    axis=1
)

# Directory to store the CSV files
temp_dir = r"\\192.168.100.252\Shared\15 - Research\000_Clean\Vinicius\Portfolio Analyzer"

# Save the final preprocessed data into CSV files
merged_data.to_csv(os.path.join(temp_dir, 'final_df.csv'), index=False)
all_news.to_csv(os.path.join(temp_dir, 'all_news.csv'), index=False)

# Ensure the directory exists
os.makedirs(temp_dir, exist_ok=True)



### ============================================== HOME Page =============================================================

st.set_page_config(page_title="Portoflio Analyzer", layout="wide")
# Display the logo at the top
st.markdown(
    '''
    <a href="https://www.unionsecurities.ch/" target="_blank">
        <img src="https://am.unionsecurities.ch/wp-content/uploads/2020/05/Logo_Union_SecuritiesWEBSITELOGOPNG.png" alt="Union Securities" width="250">
    </a>
    ''',
    unsafe_allow_html=True
    )  # Adjust the width as needed


# Introduction Section
html_home = """
   <link href='https://fonts.googleapis.com/css?family=Comfortaa' rel='stylesheet'>
<h3 style="font-family: 'Comfortaa';">Portfolio Analyzer by Union Securities Switzerland S.A. (USS)</h3>
<p style="font-family: Comfortaa;">
    The <b>Portfolio Analyzer</b> is a powerful tool designed to anaylze and construct portfolios
</p>    
    """
st.markdown(html_home, unsafe_allow_html=True)

# C:/Users/vinicius/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run "S:/15 - Research/000_Clean/Vinicius/Portfolio Analyzer/Home.py" --server.address 192.168.100.82 --server.port 8502