import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# 1. Load data
df = pd.read_csv('data/historical_data_prepro.csv')

# 2. Create Target (Next day's close price)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# 3. Define features and target
features = [
    'Open', 'High', 'Low', 'Volume',
    'Year', 'Month', 'Weekday',
    'HL_PCT', 'PCT_change', 'Avg_Price'
]
X = df[features]
y = df['Target']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocessing
# Scale only continuous numerical features
scale_cols = ['Open', 'High', 'Low', 'Volume', 'HL_PCT', 'PCT_change', 'Avg_Price']
pass_cols = ['Year', 'Month', 'Weekday']  # these are categorical/numerical but do not need scaling

preprocessor = ColumnTransformer(transformers=[
    ('scale', StandardScaler(), scale_cols),
    ('passthrough', 'passthrough', pass_cols)
])

# 6. Pipeline
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=1347, random_state=42))
])

# 7. Train
pipeline.fit(X_train, y_train)

# 8. Evaluate
preds = pipeline.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
print("RMSE:",root_mean_squared_error(y_test, preds))
print("Mean of y_test:", y_test.mean())
print("Std of y_test:", y_test.std())


# 9. Save model pipeline
joblib.dump(pipeline, 'saved_models/model.pkl')
