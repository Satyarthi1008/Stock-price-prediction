# Stock Price Prediction — Python ML Minor Project

**Level:** Intermediate  
**Market:** Indian equities (NSE)  
**Models:** Linear Regression, Random Forest Regressor, LSTM  
**Interface:** Streamlit

## 1. Project overview

This project uses historical daily OHLCV data (Open, High, Low, Close, Volume) for an NSE-listed stock. It engineers lagged/rolling indicators and predicts the next trading day's closing price. It compares three regression approaches and presents their held-out test performance in a Streamlit dashboard.

> Educational demonstration only. It is not financial advice and must not be used as the sole basis for trading or investment decisions. Stock prices are uncertain and historical patterns may not persist.

## 2. Data

Default ticker: `RELIANCE.NS` (NSE Reliance Industries). Other examples:
- `TCS.NS` — TCS
- `INFY.NS` — Infosys
- `HDFCBANK.NS` — HDFC Bank
- `ICICIBANK.NS` — ICICI Bank
- `SBIN.NS` — State Bank of India

The app downloads daily data through `yfinance` and saves a CSV in `data/`. You can also upload a CSV with `Date`, `Open`, `High`, `Low`, `Close`, and `Volume` columns. Yahoo Finance data access is provided by the community-maintained `yfinance` package; consult its documentation and applicable terms.

## 3. Setup (Windows / VS Code)

Open the project folder in VS Code, then open Terminal:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow does not install on your Python version, use a supported Python version listed by the TensorFlow installation guide (Python 3.11 is a practical starting point). Install dependencies in the same virtual environment used to run Streamlit.

## 4. Download dataset

```powershell
python download_data.py --ticker RELIANCE.NS --period 5y
```

The resulting file is saved under `data/`. Internet access is required. You may change the ticker to `TCS.NS`, `INFY.NS`, etc.

## 5. Run the dashboard

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser. Select a ticker and click **Load / refresh dataset**, then click **Train Linear Regression, Random Forest and LSTM**.

## 6. Methodology

1. Download or upload daily stock data.
2. Sort observations by date, remove invalid rows and duplicate dates.
3. Engineer daily/5-day returns, 5/10/20-day moving averages, 10-day volatility and 5-day average volume.
4. Set the target to the next available trading day's Close.
5. Split chronologically: first 80% for training, final 20% for testing.
6. Train Linear Regression and Random Forest on tabular features.
7. Train LSTM on rolling sequences of the feature history.
8. Compare MAE, RMSE and R² on the held-out test period.

The scalers are fitted on training observations only. The sequence model uses prior feature rows to estimate the next-session close. The test set is never randomly shuffled.

## 7. Evaluation metrics

- **MAE:** average absolute prediction error, in rupees.
- **RMSE:** square root of average squared error, in rupees; larger errors receive more weight.
- **R²:** proportion of target variance explained relative to a mean-baseline model. It can be negative on a poor test set.

Do not assume a model is good merely because it has a high R² on one split. Compare all models on the same test period, inspect the graph, and ideally repeat the experiment across multiple time periods.

## 8. Project structure

```text
stock_price_prediction_minor_project/
├── app.py
├── download_data.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── PRESENTATION_OUTLINE.md
├── src/
│   ├── __init__.py
│   └── stock_ml.py
├── data/       # downloaded CSV is saved here
└── outputs/    # optional exports
```

## 9. Troubleshooting

- **No data returned:** check internet access and ticker spelling; try `TCS.NS`.
- **TensorFlow import/install error:** use a compatible Python version and reinstall requirements in a fresh virtual environment.
- **CSV upload error:** check that the CSV has Date, Open, High, Low, Close and Volume fields.
- **Training is slow:** reduce LSTM epochs or use a shorter history.
- **Streamlit command not found:** activate `.venv`, then run `python -m streamlit run app.py`.

## 10. Data-source references

- yfinance documentation: https://ranaroussi.github.io/yfinance/
- yfinance download API: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- Streamlit documentation: https://docs.streamlit.io/
- scikit-learn documentation: https://scikit-learn.org/stable/
- TensorFlow documentation: https://www.tensorflow.org/

## 11. Academic note

Before submission, download the dataset, run the application, save the actual metrics and prediction chart, and replace the placeholders in `PROJECT_REPORT.md`. Do not submit invented evaluation scores.
