# MINOR PROJECT REPORT
## STOCK PRICE PREDICTION USING PYTHON AND MACHINE LEARNING

**Submitted by:** ____________________  
**Roll Number:** ____________________  
**Department / Branch:** ____________________  
**Institution:** ____________________  
**Academic Session:** ____________________

---

## Abstract

Stock price prediction is a challenging time-series problem because market prices are influenced by many changing factors. This project develops an educational machine-learning system to estimate the next trading day's closing price of an Indian NSE-listed stock using historical daily market data. The system compares Linear Regression, Random Forest Regression and Long Short-Term Memory (LSTM). Historical Open, High, Low, Close and Volume values are processed using Python. Additional features such as daily returns, moving averages and rolling volatility are calculated. The dataset is divided chronologically into training and testing portions to preserve the time order. Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and the coefficient of determination (R²). A Streamlit dashboard displays the historical price series, model comparison and prediction results. The project is intended for academic learning and does not provide investment advice.

## 1. Introduction

The stock market is a dynamic environment in which prices change in response to company performance, economic conditions, investor expectations and other information. Understanding historical price behaviour is useful for studying financial time series. Machine learning provides methods for identifying relationships in historical observations and evaluating whether those relationships help estimate later values.

The project focuses on next-session closing-price regression for an Indian stock. It implements three approaches of increasing modelling complexity: a linear baseline, a tree-ensemble model and a recurrent neural network.

## 2. Problem Statement

To design and implement a Python-based machine-learning application that processes historical daily data for an NSE-listed stock, predicts the next trading day's closing price, and compares Linear Regression, Random Forest and LSTM using a time-ordered test dataset.

## 3. Objectives

1. Collect historical Indian stock-market data.
2. Clean and preprocess daily OHLCV observations.
3. Generate useful numerical features from historical prices and volume.
4. Implement Linear Regression, Random Forest and LSTM.
5. Evaluate models using MAE, RMSE and R².
6. Visualize actual and predicted values.
7. Build an interactive Streamlit dashboard.

## 4. Scope

The project uses daily historical observations for a selected NSE-listed equity. It focuses on numerical time-series features and next-session closing-price regression. It does not include news sentiment, company financial statements, macroeconomic variables, intraday trading, portfolio optimization or automated trading.

## 5. Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Pandas, NumPy | Data manipulation and numerical processing |
| yfinance | Historical market-data download |
| Scikit-learn | Linear Regression, Random Forest, scaling and metrics |
| TensorFlow/Keras | LSTM neural network |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |

## 6. Dataset Description

The default dataset is daily historical data for Reliance Industries (`RELIANCE.NS`). The application also supports other NSE tickers such as `TCS.NS` and `INFY.NS`, and user-uploaded CSV files.

| Attribute | Description |
|---|---|
| Date | Trading date |
| Open | Opening price |
| High | Highest price during the session |
| Low | Lowest price during the session |
| Close | Closing price |
| Volume | Number of shares traded |

The data is downloaded when the user runs the application. Record the actual ticker, download date, date range and row count after downloading.

**Ticker used:** [Fill after experiment]  
**Date range:** [Fill after experiment]  
**Number of valid observations:** [Fill after experiment]

## 7. Methodology

### 7.1 Data preprocessing

The data is sorted by date, duplicate dates are removed, non-numeric and missing observations are handled, and invalid price rows are excluded.

### 7.2 Feature engineering

The input features include Open, High, Low, Close, Volume, one-day return, five-day return, 5/10/20-day moving averages, 10-day rolling volatility and five-day average volume.

The target variable is the next available trading day's closing price. This is created by shifting the Close series by one observation.

### 7.3 Train-test split

The observations are kept in chronological order. The first 80% are used for training and the final 20% for testing. Random shuffling is avoided because it could mix later observations into training.

### 7.4 Linear Regression

Linear Regression estimates a linear relationship between the engineered input features and the next-session closing price. It provides a simple baseline for comparison.

### 7.5 Random Forest Regression

Random Forest combines multiple decision trees. It can model nonlinear relationships and interactions between features. The implementation uses a fixed random seed for reproducibility.

### 7.6 LSTM

LSTM is a recurrent neural-network architecture designed to learn patterns from ordered sequences. The project supplies a rolling sequence of recent feature observations to estimate the next-session close. Feature scaling is fitted using training observations only.

## 8. System Architecture

Historical data → preprocessing → feature engineering → chronological split → model training → prediction → evaluation → Streamlit dashboard.

## 9. Evaluation Metrics

**MAE:** Mean absolute difference between actual and predicted values.  
**RMSE:** Square root of the mean squared prediction error.  
**R²:** Measures model fit relative to a mean-baseline prediction.

### Experimental results

Fill this table using the dashboard's downloaded metrics CSV. Do not enter estimated or fabricated values.

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Linear Regression | [Run project] | [Run project] | [Run project] |
| Random Forest | [Run project] | [Run project] | [Run project] |
| LSTM | [Run project] | [Run project] | [Run project] |

**Observation:** [After running, describe which model had the lowest test RMSE and discuss the actual-vs-predicted chart.]

## 10. Results and Discussion

The application produces a comparison table and a time-series chart of actual versus predicted next-session closing prices. The model with the lowest test error performed best on this particular holdout period. Performance can change with the selected stock, date range, market regime, data quality and model settings. A single holdout period is not proof of dependable future forecasting.

## 11. Limitations

1. Stock prices are affected by external events not represented in the selected features.
2. Historical price patterns may not continue in the future.
3. The project does not include news, sentiment or fundamental financial data.
4. Model performance may vary across stocks and time periods.
5. yfinance is a community-maintained interface to publicly available Yahoo Finance data and is not an official exchange feed.
6. The displayed next-session values are experimental estimates, not guaranteed prices.

## 12. Future Enhancements

- Add technical indicators such as RSI and MACD.
- Compare additional models such as Gradient Boosting and XGBoost.
- Use walk-forward validation across several test windows.
- Add more NSE stocks and index data.
- Include carefully sourced fundamental or sentiment features.
- Add model persistence and experiment tracking.

## 13. Conclusion

The project demonstrates an end-to-end machine-learning workflow for an Indian stock-price time series. It combines data collection, preprocessing, feature engineering, chronological model evaluation and an interactive dashboard. Comparing Linear Regression, Random Forest and LSTM illustrates how different modelling approaches behave on the same held-out data. The project also highlights the importance of avoiding data leakage and interpreting model metrics cautiously.

## References

1. yfinance documentation, https://ranaroussi.github.io/yfinance/
2. Scikit-learn documentation, https://scikit-learn.org/stable/
3. TensorFlow documentation, https://www.tensorflow.org/
4. Streamlit documentation, https://docs.streamlit.io/
5. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
6. S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
