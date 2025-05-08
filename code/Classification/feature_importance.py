import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('data/big_dataset.csv')

label_encoder=LabelEncoder()
df['Label']=label_encoder.fit_transform(df['debt_settlement_flag'])

df = df.drop(columns=['debt_settlement_flag'])

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Initialize LabelEncoder for each categorical   column
label_encoders = {}
for col in categorical_columns:
    df[col] = df[col].astype(str)  # Ensure all entries are strings to avoid mixed types
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into features and target
X = df.drop('Label', axis=1)  # Replace 'Label' with your target column if named differently
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LGBMClassifier with parameters
model = lgb.LGBMClassifier(
    objective='binary',  # Change to 'multiclass' if you have multiple classes
    metric='binary_logloss',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
importance_df.to_csv('data/feature_importance.csv')

# Making filtered Dataset
IMPORTANCE_THRESHOLD = 95
important_features = importance_df[importance_df['Importance'] >= IMPORTANCE_THRESHOLD]
print(f"Selected {len(important_features)} features:")
print(important_features)

selected_columns = important_features['Feature'].tolist() + ['Label']
final_df = df[selected_columns]
final_df.to_csv('data/filtered_dataset.csv', index=False)
print(f"\nSaved filtered dataset with {len(selected_columns)} columns to data/filtered_dataset.csv")