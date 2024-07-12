import joblib
import pandas as pd

# Load the pre-trained model
model_path = 'autots_model.pkl'
try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Test input data
test_data = {
    'lag_1': 1000.0,
    'lag_2': 1000.0,
    'lag_3': 1000.0,
    'rolling_mean_3': 1000.0,
    'rolling_std_3': 1000.0,
    'quarter': 1,
    'day_of_week': 0,
    'is_weekend': 0,
    'ema_3': 1000.0,
    'autocorr_1': 0.0,
    'autocorr_2': 0.0
}
input_df = pd.DataFrame([test_data])

# Print the input DataFrame
print(f"Input DataFrame:\n{input_df}")
print(f"Input DataFrame shape: {input_df.shape}")

# Make prediction
try:
    prediction = model.predict(input_df)
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error during prediction: {e}")