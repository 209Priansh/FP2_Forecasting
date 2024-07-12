import streamlit as st
import requests
from datetime import datetime

# Define the FastAPI endpoint URL
endpoint = 'http://127.0.0.1:8000/predict'

def main():
    st.markdown("# Petroleum Production Forecasting")
    st.sidebar.title("Petroleum Dashboard")
    
    # Input fields for user input
    st.header('Input Data')
    lag_1 = st.number_input('lag_1', value=946.28)
    lag_2 = st.number_input('lag_2', value=942.43)
    lag_3 = st.number_input('lag_3', value=967.75)
    rolling_mean_3 = st.number_input('rolling_mean_3', value=938.38)
    rolling_std_3 = st.number_input('rolling_std_3', value=10.52)
    quarter = st.number_input('quarter', min_value=1, max_value=4, value=2)
    day_of_week = st.number_input('day_of_week', min_value=0, max_value=6, value=1)
    is_weekend = st.selectbox('is_weekend', ['0', '1'])
    ema_3 = st.number_input('ema_3', value=938.56)
    autocorr_1 = st.number_input('autocorr_1', value=0)
    autocorr_2 = st.number_input('autocorr_2', value=0)

    # Input field for period/date
    period = st.date_input('Prediction Period', value=datetime.today())

    # Button to trigger prediction
    if st.button('Predict'):
        # Prepare data payload
        payload = {
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_3,
            'rolling_mean_3': rolling_mean_3,
            'rolling_std_3': rolling_std_3,
            'quarter': quarter,
            'day_of_week': day_of_week,
            'is_weekend': int(is_weekend),
            'ema_3': ema_3,
            'autocorr_1': autocorr_1,
            'autocorr_2': autocorr_2,
            'period': period.strftime('%Y-%m-%d')  # Convert period to string format
        }

        # Make POST request to FastAPI endpoint
        try:
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                prediction = response.json()['prediction']
                st.success(f'Prediction for {period.strftime("%Y-%m-%d")}: {prediction}')
            else:
                st.error(f'Failed with status code: {response.status_code}')
        except Exception as e:
            st.error(f'Error occurred: {e}')

if __name__ == '__main__':
    main()
