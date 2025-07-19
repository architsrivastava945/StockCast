import yfinance as yf
import pandas as pd

def fetch_nifty50_data():
    # Download historical data for Nifty 50 (symbol: ^NSEI)
    df = yf.download('^NSEI', start='2000-01-01', end='2024-12-31', interval='1d')
    
    # Save to CSV inside /data folder
    df.to_csv('data/historical_data.csv')
    print("✅ Data saved to data/historical_data.csv")

if __name__ == "__main__":
    fetch_nifty50_data()
