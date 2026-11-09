from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Yahoo/Kaggle CSV column formats."""
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).strip().title().replace("Adj Close", "Adj Close") for c in data.columns]

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.set_index("Date")
    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")

    rename = {}
    for col in data.columns:
        key = str(col).strip().lower().replace("_", " ")
        if key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.title()
        elif key in {"adj close", "adjclose", "adjusted close"}:
            rename[col] = "Adj Close"
    data = data.rename(columns=rename)

    missing = [c for c in BASE_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    data = data[BASE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = data[~data.index.isna()].sort_index()
    data = data[~data.index.duplicated(keep="last")].dropna()
    data = data[(data["Close"] > 0) & (data["Volume"] >= 0)]
    if len(data) < 180:
        raise ValueError("At least 180 valid daily rows are recommended; this dataset has fewer.")
    data.index.name = "Date"
    return data


def download_stock_data(ticker: str = "RELIANCE.NS", period: str = "5y") -> pd.DataFrame:
    """Download daily NSE data and save a reusable CSV copy."""
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the symbol or try later.")
    data = normalize_ohlcv(raw)
    safe_ticker = ticker.replace(".", "_")
    data.to_csv(DATA_DIR / f"{safe_ticker}_{period}.csv")
    return data


def load_uploaded_csv(file_obj) -> pd.DataFrame:
    data = pd.read_csv(file_obj)
    return normalize_ohlcv(data)


def prepare_features(df: pd.DataFrame):
    """Create features known at the end of day t to predict Close at t+1."""
    data = normalize_ohlcv(df)
    x = pd.DataFrame(index=data.index)
    x["Open"] = data["Open"]
    x["High"] = data["High"]
    x["Low"] = data["Low"]
    x["Close"] = data["Close"]
    x["Volume"] = data["Volume"]
    x["Return_1d"] = data["Close"].pct_change()
    x["Return_5d"] = data["Close"].pct_change(5)
    x["MA_5"] = data["Close"].rolling(5).mean()
    x["MA_10"] = data["Close"].rolling(10).mean()
    x["MA_20"] = data["Close"].rolling(20).mean()
    x["Volatility_10"] = x["Return_1d"].rolling(10).std()
    x["Volume_MA_5"] = data["Volume"].rolling(5).mean()

    # Target is next trading day's close. Shift prevents using future target as input.
    y = data["Close"].shift(-1).rename("Target_Next_Close")
    joined = x.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    return joined.drop(columns=["Target_Next_Close"]), joined["Target_Next_Close"]


def _metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def train_and_evaluate(df: pd.DataFrame, sequence_length: int = 30, epochs: int = 15):
    """Chronological 80/20 split; all scalers are fitted on training data only."""
    X, y = prepare_features(df)
    split = int(len(X) * 0.80)
    if split <= sequence_length or len(X) - split < 20:
        raise ValueError("Not enough data after feature engineering for this split.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    models, metric_rows = {}, []
    predictions = pd.DataFrame(index=y_test.index)
    predictions["Actual"] = y_test

    lr = make_pipeline(StandardScaler(), LinearRegression())
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models["Linear Regression"] = lr
    predictions["Linear Regression"] = lr_pred
    metric_rows.append({"Model": "Linear Regression", **_metrics(y_test, lr_pred)})

    rf = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models["Random Forest"] = rf
    predictions["Random Forest"] = rf_pred
    metric_rows.append({"Model": "Random Forest", **_metrics(y_test, rf_pred)})

    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError("TensorFlow is required for LSTM. Install requirements.txt and retry.") from exc

    # Scale using training rows only, then create sequences ending at each target row.
    scaler = StandardScaler()
    scaler.fit(X.iloc[:split])
    X_scaled = scaler.transform(X)

    seq_x, seq_y, seq_dates, target_positions = [], [], [], []
    values = y.to_numpy()
    for end_pos in range(sequence_length - 1, len(X)):
        seq_x.append(X_scaled[end_pos-sequence_length+1:end_pos+1])
        seq_y.append(values[end_pos])
        seq_dates.append(y.index[end_pos])
        target_positions.append(end_pos)

    seq_x = np.asarray(seq_x, dtype=np.float32)
    seq_y = np.asarray(seq_y, dtype=np.float32)
    target_positions = np.asarray(target_positions)
    train_mask = target_positions < split
    test_mask = target_positions >= split
    X_seq_train, y_seq_train = seq_x[train_mask], seq_y[train_mask]
    X_seq_test, y_seq_test = seq_x[test_mask], seq_y[test_mask]
    test_dates = pd.DatetimeIndex(np.asarray(seq_dates)[test_mask])

    # Deterministic seeds improve reproducibility (hardware can still affect results).
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    lstm = Sequential([
        LSTM(48, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
        Dropout(0.15),
        LSTM(24),
        Dropout(0.15),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mean_squared_error")
    stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    lstm.fit(X_seq_train, y_seq_train, validation_split=0.15, shuffle=False,
             epochs=epochs, batch_size=32, callbacks=[stop], verbose=0)

    lstm_pred = lstm.predict(X_seq_test, verbose=0).reshape(-1)
    predictions.loc[test_dates, "LSTM"] = lstm_pred
    # Align metrics exactly to LSTM's test dates.
    lstm_actual = y.loc[test_dates]
    metric_rows.append({"Model": "LSTM", **_metrics(lstm_actual, lstm_pred)})

    predictions = predictions.dropna()
    metrics = pd.DataFrame(metric_rows).set_index("Model").sort_values("RMSE")
    return {
        "metrics": metrics,
        "predictions": predictions,
        "models": models,
        "lstm": lstm,
        "feature_scaler": scaler,
        "feature_columns": list(X.columns),
        "sequence_length": sequence_length,
        "split": split,
        "last_features": X,
    }


def next_day_estimate(df: pd.DataFrame, results: dict, sequence_length: int = 30):
    """Experimental one-step estimate from the most recent available sequence."""
    X, _ = prepare_features(df)
    X = X[results["feature_columns"]]
    estimates = {}
    last_row = X.iloc[[-1]]
    for name in ["Linear Regression", "Random Forest"]:
        estimates[name] = float(results["models"][name].predict(last_row)[0])

    scaled = results["feature_scaler"].transform(X.tail(sequence_length))
    seq = scaled.reshape(1, sequence_length, X.shape[1]).astype(np.float32)
    estimates["LSTM"] = float(results["lstm"].predict(seq, verbose=0)[0, 0])
    return estimates
