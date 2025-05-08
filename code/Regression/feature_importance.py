import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
print("Loading data...")
df = pd.read_csv('data/defaulters.csv')

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Initialize LabelEncoder for each categorical   column
label_encoders = {}
for col in categorical_columns:
    df[col] = df[col].astype(str)  # Ensure all entries are strings to avoid mixed types
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Define features and target
print("Preparing features and target...")
X = df.drop(['settlement_amount'], axis=1)
y = df['settlement_amount'] 

# Initialize and fit LightGBM
print("Fitting LightGBM...")
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X, y)

# Predictions
y_pred = lgb_model.predict(X)

# Feature importance
importance = lgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)

# Feature importance
importance = lgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
importance_df.to_csv('data/feature_importance.csv')

# Making filtered Dataset
IMPORTANCE_THRESHOLD = 68
important_features = importance_df[importance_df['Importance'] >= IMPORTANCE_THRESHOLD]
print(f"Selected {len(important_features)} features:")
print(important_features)

selected_columns = important_features['Feature'].tolist() + ['settlement_amount']
final_df = df[selected_columns]
final_df.to_csv('data/filtered_defaulters.csv', index=False)
print(f"\nSaved filtered dataset with {len(selected_columns)} columns to data/filtered_defaulters.csv")