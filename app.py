from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.stock_ml import (
    download_stock_data, load_uploaded_csv, prepare_features,
    train_and_evaluate, next_day_estimate
)

st.set_page_config(page_title="Indian Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Indian Stock Price Prediction")
st.caption("Educational Machine Learning project | Linear Regression • Random Forest • LSTM")

with st.sidebar:
    st.header("Dataset settings")
    ticker = st.selectbox(
        "NSE ticker",
        ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS"],
        index=0
    )
    period = st.selectbox("Historical period", ["2y", "5y", "10y"], index=1)
    window = st.slider("LSTM sequence length (trading days)", 10, 60, 30, 5)
    epochs = st.slider("LSTM training epochs", 5, 50, 15, 5)
    uploaded = st.file_uploader("Or upload a CSV dataset", type=["csv"])
    load_button = st.button("Load / refresh dataset", type="primary", use_container_width=True)

if "stock_df" not in st.session_state:
    st.session_state.stock_df = None
    st.session_state.ticker = ticker

if load_button:
    try:
        with st.spinner("Loading historical data..."):
            if uploaded is not None:
                df = load_uploaded_csv(uploaded)
                st.session_state.ticker = "Uploaded CSV"
            else:
                df = download_stock_data(ticker, period)
                st.session_state.ticker = ticker
            st.session_state.stock_df = df
            st.session_state.results = None
        st.success(f"Loaded {len(df):,} daily observations.")
    except Exception as exc:
        st.error(f"Could not load data: {exc}")

if st.session_state.stock_df is None:
    st.info("Choose an NSE ticker and click **Load / refresh dataset**, or upload a CSV with Date, Open, High, Low, Close and Volume columns.")
    st.markdown("""
    **Suggested first run:** `RELIANCE.NS`, 5 years.

    The app fetches historical data using `yfinance`. Internet access is required for downloading.
    """)
    st.stop()

df = st.session_state.stock_df.copy()
st.subheader(f"Dataset: {st.session_state.ticker}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}")
c2.metric("First date", str(df.index.min().date()))
c3.metric("Latest date", str(df.index.max().date()))
c4.metric("Latest close", f"₹{df['Close'].iloc[-1]:,.2f}")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.72, 0.28])
fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low,
                             close=df.Close, name="Price"), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume"), row=2, col=1)
fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Train and evaluate models")
st.write("The last 20% of observations are held out chronologically for testing. No random shuffling is used.")
if st.button("Train Linear Regression, Random Forest and LSTM", type="primary"):
    try:
        with st.spinner("Training models. LSTM training can take a few minutes..."):
            results = train_and_evaluate(df, sequence_length=window, epochs=epochs)
            st.session_state.results = results
        st.success("Training and evaluation completed.")
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.exception(exc)

results = st.session_state.get("results")
if results:
    st.subheader("Test-set evaluation")
    st.dataframe(results["metrics"].style.format({
        "MAE": "₹{:.2f}", "RMSE": "₹{:.2f}", "R2": "{:.4f}"
    }), use_container_width=True)
    st.caption("Lower MAE/RMSE and higher R² generally indicate better fit on this held-out period. They do not guarantee future performance.")

    pred = results["predictions"]
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=pred.index, y=pred["Actual"], name="Actual close"))
    for name in [c for c in pred.columns if c != "Actual"]:
        pfig.add_trace(go.Scatter(x=pred.index, y=pred[name], name=name))
    pfig.update_layout(title="Actual vs predicted next-day closing price", height=480,
                       yaxis_title="Price (₹)", xaxis_title="Date")
    st.plotly_chart(pfig, use_container_width=True)

    st.download_button("Download evaluation predictions CSV",
                       pred.to_csv().encode("utf-8"),
                       file_name="stock_model_predictions.csv", mime="text/csv")
    st.download_button("Download model metrics CSV",
                       results["metrics"].to_csv().encode("utf-8"),
                       file_name="stock_model_metrics.csv", mime="text/csv")

    st.subheader("Next-session model estimates")
    latest = next_day_estimate(df, results, sequence_length=window)
    cols = st.columns(len(latest))
    for col, (model_name, value) in zip(cols, latest.items()):
        col.metric(model_name, f"₹{value:,.2f}")
    st.warning("These are experimental model estimates, not reliable forecasts or investment recommendations. Actual prices can differ substantially.")

with st.expander("View raw dataset"):
    st.dataframe(df.tail(30), use_container_width=True)
