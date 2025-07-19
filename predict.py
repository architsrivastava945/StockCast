# predict.py

import joblib
import pandas as pd

# Load the trained model
model = joblib.load('saved_models/model.pkl')

# Define the columns in correct order
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Volume',
    'Year', 'Month', 'Weekday',
    'HL_PCT', 'PCT_change', 'Avg_Price'
]

def predict_close_price(input_data):
    """
    Predict stock close price given input data.
    
    Parameters:
    input_data (dict): Dictionary containing values for required features
    
    Returns:
    float: Predicted close price
    """
    # Convert input dict to DataFrame with correct column order
    df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    
    # Predict
    prediction = model.predict(df)[0]
    
    return prediction

# Example usage for testing (will only run if you directly run predict.py)
if __name__ == "__main__":
    sample_input = {
        'Open': 105.0,
        'High': 110.0,
        'Low': 100.0,
        'Volume': 1000000,
        'Year': 2023,
        'Month': 7,
        'Weekday': 2,  # 0=Monday, 6=Sunday
        'HL_PCT': (110.0 - 100.0) / 100.0 * 100,
        'PCT_change': (105.0 - 100.0) / 100.0 * 100,
        'Avg_Price': (105.0 + 110.0 + 100.0) / 3
    }
    result = predict_close_price(sample_input)
    print(f"Predicted Close Price: ₹{result:.2f}")
