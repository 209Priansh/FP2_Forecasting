import pandas as pd

def preprocess_and_engineer_features(data):
     # Convert index to datetime if it's not already
    data.index = pd.to_datetime(data.index)
    data['lag_1'] = data['value'].shift(1)
    data['lag_2'] = data['value'].shift(2)
    data['lag_3'] = data['value'].shift(3)
    data['rolling_mean_3'] = data['value'].rolling(window=3).mean()
    data['rolling_std_3'] = data['value'].rolling(window=3).std()
    data['quarter'] = data.index.quarter
    data['day_of_week'] = data.index.dayofweek
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['ema_3'] = data['value'].ewm(span=3, adjust=False).mean()
    data['autocorr_1'] = data['value'].rolling(window=12).apply(lambda x: x.autocorr(lag=1))
    data['autocorr_2'] = data['value'].rolling(window=12).apply(lambda x: x.autocorr(lag=2))
    data=data.dropna()
    return data


