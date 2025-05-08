import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the data
print("Loading data...")
df = pd.read_csv('data/filtered_dataset.csv')

# Separate features and target
print("Preparing features and target...")
X = df.drop('Label', axis=1)
y = df['Label']

# First split - create a holdout set
print("Creating holdout set...")
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Handle missing values using SimpleImputer
print("Handling missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_temp)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Split remaining data into train and validation
print("Splitting data into train and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

print("\nOriginal class distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True))

# Apply SMOTE
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42, sampling_strategy=1.0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nResampled class distribution in training set:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)

# Initialize and train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train_scaled, y_train_resampled)

# Evaluation
print("\nEvaluating model...")
# Validation set
val_preds = rf_model.predict(X_val_scaled)
val_probs = rf_model.predict_proba(X_val_scaled)

print("\nValidation Set Performance:")
print(classification_report(y_val, val_preds))

# Holdout set
holdout_preds = rf_model.predict(X_holdout_scaled)
holdout_probs = rf_model.predict_proba(X_holdout_scaled)

print("\nHoldout Set Performance:")
print(classification_report(y_holdout, holdout_preds))

# Print sample probabilities
print("\nSample probabilities for first 5 examples (Holdout set):")
print("Class 0 prob | Class 1 prob")
# Save probabilities to CSV and display first 10 rows
probs_df = pd.DataFrame(holdout_probs, columns=['Class_0_Prob', 'Class_1_Prob'])
probs_df['Actual_Class'] = y_holdout.values
probs_df.to_csv('data/holdout_probabilities.csv', index=False)
print(holdout_probs[:10])

import joblib
joblib.dump(rf_model, 'random_forest_model.joblib')
print("Model saved successfully!")

# Create confusion matrix for holdout set
print("\nCreating confusion matrix plot...")
cm = confusion_matrix(y_holdout, holdout_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Holdout Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.show()

print("Analysis complete!")