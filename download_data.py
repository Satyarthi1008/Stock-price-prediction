import argparse
from src.stock_ml import download_stock_data

parser = argparse.ArgumentParser(description="Download daily NSE stock history.")
parser.add_argument("--ticker", default="RELIANCE.NS", help="Yahoo Finance NSE ticker, e.g. TCS.NS")
parser.add_argument("--period", default="5y", choices=["2y", "5y", "10y", "max"])
args = parser.parse_args()

df = download_stock_data(args.ticker, args.period)
print(f"Saved {len(df)} rows for {args.ticker} to data/")
print(df.tail())
