import pandas as pd
import joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

MODEL_PATH = "sales_model.joblib"

def prepare_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    if 'is_promo' not in df.columns: df['is_promo'] = 0
    df = df.sort_values('date')
    df['lag_1'] = df['sales'].shift(1).fillna(method='bfill')
    df['lag_7'] = df['sales'].shift(7).fillna(method='bfill')
    df.fillna(0, inplace=True)
    X = df[['dayofweek','day','month','weekofyear','is_promo','lag_1','lag_7']]
    y = df['sales']
    return X, y

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    return {"mae": mae}

def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def predict_next_days(df, n_days=14):
    model = load_model()
    if model is None: raise ValueError("Train model first")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    hist = df.sort_values('date').copy()
    future = []
    for i in range(1, n_days+1):
        next_date = last_date + pd.Timedelta(days=i)
        row = pd.DataFrame([{
            'dayofweek': next_date.dayofweek,
            'day': next_date.day,
            'month': next_date.month,
            'weekofyear': next_date.isocalendar()[1],
            'is_promo': 0,
            'lag_1': hist['sales'].iloc[-1],
            'lag_7': hist['sales'].iloc[-7] if len(hist) >= 7 else hist['sales'].iloc[0]
        }])
        pred = model.predict(row)[0]
        future.append({"date": next_date.date(), "predicted_sales": round(pred,2)})
        hist = hist.append({'date': next_date, 'sales': pred}, ignore_index=True)
    return future
