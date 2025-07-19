# app.py

import streamlit as st
from predict import predict_close_price
import pandas as pd

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

# Title
st.title("📈 Stock Close Price Predictor")
st.markdown("This app uses a trained Random Forest model to predict the **closing price** of a stock based on input features.")

# Input form
with st.form("prediction_form"):
    st.subheader("🔢 Enter Stock Features")

    col1, col2 = st.columns(2)

    with col1:
        open_price = st.number_input("Open Price", value=100.0, step=1.0)
        high_price = st.number_input("High Price", value=105.0, step=1.0)
        low_price = st.number_input("Low Price", value=95.0, step=1.0)
        volume = st.number_input("Volume", value=1000000, step=10000)

    with col2:
        year = st.number_input("Year", value=2024, step=1, format="%d")
        month = st.number_input("Month", min_value=1, max_value=12, value=7, step=1)
        weekday = st.selectbox("Weekday", options=[
            (0, "Monday"), (1, "Tuesday"), (2, "Wednesday"),
            (3, "Thursday"), (4, "Friday"), (5, "Saturday"), (6, "Sunday")
        ], format_func=lambda x: x[1])[0]

    # Automatically calculate derived features
    hl_pct = ((high_price - low_price) / low_price) * 100
    pct_change = ((open_price - low_price) / low_price) * 100
    avg_price = (open_price + high_price + low_price) / 3

    st.markdown(f"**🧮 Derived Features:**")
    st.write(f"- HL %: {hl_pct:.2f}")
    st.write(f"- % Change: {pct_change:.2f}")
    st.write(f"- Avg Price: {avg_price:.2f}")

    # Submit
    submitted = st.form_submit_button("Predict 📊")

if submitted:
    # Prepare input for prediction
    input_data = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Volume": volume,
        "Year": year,
        "Month": month,
        "Weekday": weekday,
        "HL_PCT": hl_pct,
        "PCT_change": pct_change,
        "Avg_Price": avg_price
    }

    # Get prediction
    prediction = predict_close_price(input_data)

    # Display result
    st.success(f"📌 Predicted Close Price: ₹{prediction:.2f}")
