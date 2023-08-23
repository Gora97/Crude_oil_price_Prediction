import streamlit as st
import pandas as pd
from joblib import load
from prophet import Prophet
import plotly as px
import os
# Construct the path to your model
model_filename = os.path.join(os.path.dirname(__file__), 'models', 'prophet_model.joblib')

# Load the saved Prophet model using joblib
model = load(model_filename)

st.title('Oil Price Prediction App')

# Page 1 Layout
st.markdown('## Predict Oil Prices')
selected_date = st.date_input('Select a date', pd.to_datetime('2023-08-10').date())
predict_button = st.button('Predict Price')

if predict_button:
    future = pd.DataFrame({'ds': [selected_date]})
    forecast = model.predict(future)
    predicted_price = forecast.loc[0, 'yhat']
    
    st.markdown(f"The predicted price on {selected_date} is {predicted_price:.2f}$")

# Page 2 Layout
st.markdown('## Forecast Prices')
selected_duration = st.selectbox('Select Forecast Duration', [1, 2, 3, 5, 10])
start_date = pd.Timestamp('2023-01-01')
end_date = start_date + pd.DateOffset(days=365 * selected_duration)

future = pd.DataFrame({'ds': pd.date_range(start_date, end_date)})
forecast = model.predict(future)
forecast.rename(columns={'ds': 'Date', 'yhat': 'Price'}, inplace=True)

st.plotly_chart(px.line(forecast, x='Date', y='Price', title=f'Forecasted Prices for {selected_duration} Years'))

st.markdown('[Go to Page 1](#predict-oil-prices)')
