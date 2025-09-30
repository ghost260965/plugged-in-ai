# predictor.py — SKU-level forecasting
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

MODEL_DIR = "models"   # store one model per SKU
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    if 'is_promo' not in df.columns:
        df['is_promo'] = 0
    df = df.sort_values('date')
    df['lag_1'] = df['sales'].shift(1).fillna(method='bfill')
    df['lag_7'] = df['sales'].shift(7).fillna(method='bfill')
    df.fillna(0, inplace=True)
    X = df[['dayofweek','day','month','weekofyear','is_promo','lag_1','lag_7']]
    y = df['sales']
    return X, y

def train_models(csv_path, sku_col="product_id"):
    df = pd.read_csv(csv_path)
    if sku_col not in df.columns:
        raise ValueError(f"CSV must contain a '{sku_col}' column for SKU-level forecasting.")
    
    results = {}
    for sku, subdf in df.groupby(sku_col):
        if len(subdf) < 14:  # need enough history
            continue
        X, y = prepare_features(subdf)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        model_path = os.path.join(MODEL_DIR, f"model_{sku}.joblib")
        joblib.dump(model, model_path)
        results[sku] = {"mae": float(mae), "model_path": model_path}
    return results

def load_model(sku):
    path = os.path.join(MODEL_DIR, f"model_{sku}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def predict_next_days(df, sku, n_days=14):
    model = load_model(sku)
    if model is None:
        raise ValueError(f"No trained model found for SKU {sku}.")
    df = df[df['product_id'] == sku].copy()
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    hist = df.sort_values('date').copy()
    future = []

    for i in range(1, n_days+1):
        next_date = last_date + pd.Timedelta(days=i)
        row = {
            'dayofweek': next_date.dayofweek,
            'day': next_date.day,
            'month': next_date.month,
            'weekofyear': next_date.isocalendar()[1],
            'is_promo': 0
        }
        lag_1 = hist['sales'].iloc[-1]
        lag_7 = hist['sales'].iloc[-7] if len(hist) >= 7 else hist['sales'].iloc[0]
        Xrow = pd.DataFrame([{
            'dayofweek': row['dayofweek'],
            'day': row['day'],
            'month': row['month'],
            'weekofyear': row['weekofyear'],
            'is_promo': row['is_promo'],
            'lag_1': lag_1,
            'lag_7': lag_7
        }])
        pred = model.predict(Xrow)[0]
        future.append({"date": next_date.date(), "predicted_sales": float(round(pred,2))})
        hist = hist.append({'date': next_date, 'sales': pred}, ignore_index=True)
    return future

def top_products_forecast(csv_path, n_days=14, top_k=3):
    df = pd.read_csv(csv_path)
    results = []
    for sku in df['product_id'].unique():
        try:
            fut = predict_next_days(df, sku, n_days)
            avg_sales = np.mean([x['predicted_sales'] for x in fut])
            results.append({"product_id": sku, "avg_predicted_sales": avg_sales})
        except Exception:
            continue
    results = sorted(results, key=lambda x: x['avg_predicted_sales'], reverse=True)
    return results[:top_k]

