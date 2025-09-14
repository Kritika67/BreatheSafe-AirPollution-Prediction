import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Model Loading ---
# Check if the model file exists before loading
if not os.path.exists('air_pollution_xgb_model.pkl'):
    st.error("Model file 'air_pollution_xgb_model.pkl' not found. Please run your backend script first to train and save the model.")
    st.stop()
if not os.path.exists('model_features.pkl'):
    st.error("Model features file 'model_features.pkl' not found. Please run your backend script first.")
    st.stop()
try:
    model = joblib.load('air_pollution_xgb_model.pkl')
    all_features = joblib.load('model_features.pkl')
except Exception as e:
    st.error(f"Error loading model or feature list: {e}")
    st.stop()

# --- Application UI ---
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2e8b57;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #2e8b57;
        color: white;
        font-size: 1.2rem;
        border-radius: 10px;
        border: 2px solid #2e8b57;
        display: block;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
        color: black;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e8b57;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("BreatheSafe: Air Pollution Predictor")
st.write("Enter the following details to predict the Air Quality Index (AQI) for a specific location.")
st.markdown("---")

# Main prediction interface
col1, col2 = st.columns(2)
user_input_data = {}
with col1:
    st.subheader("Pollution Data")
    user_input_data['PM2.5'] = st.number_input("PM2.5", value=50.0)
    user_input_data['PM10'] = st.number_input("PM10", value=50.0)
    user_input_data['NO2'] = st.number_input("NO2", value=50.0)
    user_input_data['SO2'] = st.number_input("SO2", value=50.0)
    user_input_data['O3'] = st.number_input("O3", value=50.0)
    user_input_data['CO'] = st.number_input("CO", value=50.0)
    
with col2:
    st.subheader("Time & Location")
    user_input_data['AQI_rolling_avg'] = st.number_input("3-Day Avg AQI", value=50.0)
    user_input_data['day_of_week'] = st.selectbox("Day of the Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
    user_input_data['month'] = st.selectbox("Month", range(1, 13))
    user_input_data['year'] = st.selectbox("Year", [2024, 2025])
    user_input_data['PM2.5_lag1'] = st.number_input("PM2.5 (1 day ago)", value=50.0)
    user_input_data['PM2.5_lag7'] = st.number_input("PM2.5 (7 days ago)", value=50.0)
    user_input_data['PM2.5_rolling_mean_7d'] = st.number_input("PM2.5 (7-day mean)", value=50.0)
    selected_city = st.selectbox("Select a City", ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Gandhinagar', 'Gurugram', 'Guwahati', 'Hyderabad', 'Indore', 'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai', 'Nagpur', 'Patna', 'Shillong', 'Thiruvananthapuram', 'Visakhapatnam'])

st.markdown("---")

if st.button("Predict AQI"):
    # Create a DataFrame for prediction
    final_input_data = {feature: [0.0] for feature in all_features}
    
    # Fill in the user-provided data
    for feature, value in user_input_data.items():
        if feature in final_input_data:
            final_input_data[feature] = [value]
            
    final_input_data[f"city_{selected_city}"] = [1.0]

    input_df = pd.DataFrame(final_input_data, columns=all_features)
    
    try:
        predicted_aqi = model.predict(input_df)[0]
        
        st.subheader("Prediction Result")
        st.markdown(f"**The predicted AQI for {selected_city} is: <span style='color:green; font-size: 2rem;'>{predicted_aqi:.2f}</span>**", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed. Please check the input data and try again. Error: {e}")






