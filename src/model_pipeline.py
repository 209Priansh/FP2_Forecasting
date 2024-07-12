import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from autots import AutoTS
import joblib
import mlflow
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

# Load your data (assuming it's in Excel format)
data_file_path = r'C:\Prashant\ISB\TERM5\FP2\ASSIGN\Code\pre1_india_petroleum_data.xlsx'
df = pd.read_excel(data_file_path)

# Function to preprocess and engineer features
def preprocess_and_engineer_features(data):
    data.set_index('period', inplace=True)
    
    # Feature engineering
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
    data = data.dropna()  # Drop rows with NaN values
    return data

# Preprocess the data
data = preprocess_and_engineer_features(df)
print(data.head())

# Decompose the time series if there are enough observations
if len(data) >= 24:  # Adjust threshold as needed
    decomposition = seasonal_decompose(data['value'], model='multiplicative', period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot decomposition
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(data['value'], label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for seasonal decomposition.")

# Correlation Analysis
numeric_columns = ['value', 'lag_1', 'lag_2', 'lag_3',
                   'rolling_mean_3', 'rolling_std_3', 'is_weekend', 'ema_3',
                   'autocorr_1', 'autocorr_2']

plt.figure(figsize=(10, 8))
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.01)
data['anomaly'] = iso_forest.fit_predict(data[['value']])

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Production')
plt.scatter(data[data['anomaly'] == -1].index, data[data['anomaly'] == -1]['value'], color='red', label='Anomaly')
plt.title('Anomaly Detection in Petroleum Production')
plt.xlabel('Date')
plt.ylabel('Production (TBPD)')
plt.legend()
plt.show()

# Define features and target
features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'quarter', 'day_of_week', 'is_weekend', 'ema_3', 'autocorr_1', 'autocorr_2']
X = data[features]
y = data['value']

# Split data considering recent data at the top
y_test = y.iloc[:72]    # Take the top 72 months for testing/validation (most recent)
y_train = y.iloc[72:]   # Take the remaining months for training (older)
X_test = X.iloc[:72]    # Features corresponding to y_test
X_train = X.iloc[72:]   # Features corresponding to y_train

# Perform Standard Scaling on features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'y_train size: {len(y_train)}')
print(f'y_test size: {len(y_test)}')
print(f'X_train size: {len(X_train)}')
print(f'X_test size: {len(X_test)}')

# Determine forecast length dynamically
forecast_length = min(72, len(y_train))  # Forecast length set to 72 months
print(f"Forecast Length: {forecast_length}")

# Initialize AutoTS
model = AutoTS(forecast_length=forecast_length, frequency='ME', max_generations=5, drop_most_recent=1, num_validations=3)

# Fit AutoTS with preprocessed data
try:
    model.fit(y_train)
    forecasts = model.predict()
    forecast_values = forecasts.forecast

    # Save the model using joblib
    joblib.dump(model, 'autots_model.pkl')

    # Define MAPE function
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate RMSE, MAE, and MAPE
    y_pred = forecast_values.values.flatten()[:len(y_test)]
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")

    # Log the saved model and metrics using mlflow
    with mlflow.start_run():
        mlflow.log_artifact('autots_model.pkl')
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAPE", mape)

    print(f"Model RMSE: {rmse}")
    print(f"Model MAE: {mae}")
    print(f"Model MAPE: {mape}")

except ValueError as e:
    print(f"Error fitting AutoTS model: {e}")

# End the current MLflow run
mlflow.end_run()

# Predict future values
future_forecast = model.predict().forecast

# Visualize the predictions along with the actual values
plt.figure(figsize=(12, 6))
plt.plot(data.index[-72:], y_test, label='Actual')
plt.plot(data.index[-72:], y_pred, label='Forecast')
plt.title('Actual vs Forecasted Petroleum Production')
plt.xlabel('Date')
plt.ylabel('Production (TBPD)')
plt.legend()
plt.show()

# Predict future values beyond the test set
future_dates = pd.date_range(start=data.index[0], periods=forecast_length + 1, freq='ME')[1:]
future_forecast_values = future_forecast.values.flatten()

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'period': future_dates, 'predicted_value': future_forecast_values})

# Save future forecast to Excel
#output_file_path = r'C:\Prashant\ISB\TERM5\FP2\ASSIGN\Code\future_forecast.xlsx'
#future_df.to_excel(output_file_path, index=False)

print(future_df)
