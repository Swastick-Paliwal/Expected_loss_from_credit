import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Read the data
df = pd.read_csv('data/filtered_defaulters.csv')

# Assuming we need to predict settlement_amount
# Drop any rows with missing values
df = df.dropna()

# Split features (X) and target (y)
# Exclude settlement_amount from features
X = df.drop('settlement_amount', axis=1)
y = df['settlement_amount']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Coefficient', key=abs, ascending=False))

# Save the model and feature importance
print("\nSaving model...")
joblib.dump(model, 'regression_model.joblib')
print("Model and feature importance saved successfully!")