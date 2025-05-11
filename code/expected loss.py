import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

regression_features = 'installment,loan_amnt,int_rate,term,dti,total_bc_limit,mo_sin_old_il_acct,sub_grade,revol_util,bc_open_to_buy'

# Load the trained models
regression_model = joblib.load('data/regression_model.joblib')
classification_model = joblib.load('data/classification_model.joblib')

# Load the dataset
df_classification = pd.read_csv('data/filtered_dataset.csv')
classification_features = df_classification.drop('Label', axis=1).columns

# Ensure classification features are in the same order as training
df_classification = df_classification[classification_features]

# Load regression dataset with explicit feature order
regression_features_list = regression_features.split(',')
df_regression = pd.read_csv('data/big_dataset.csv', usecols=regression_features_list)

# Identify categorical columns
categorical_columns = df_regression.select_dtypes(include=['object']).columns.tolist()

# Initialize LabelEncoder for each categorical   column
label_encoders = {}
for col in categorical_columns:
    df_regression[col] = df_regression[col].astype(str)  # Ensure all entries are strings to avoid mixed types
    le = LabelEncoder()
    df_regression[col] = le.fit_transform(df_regression[col])
    label_encoders[col] = le

# Ensure correct feature order
df_regression = df_regression[regression_features_list]

# Drop any rows with missing values
df_regression = df_regression.dropna()

# Get probability of default (class 1) from classification model
default_probabilities = classification_model.predict_proba(df_classification)[:, 1]

# Get predicted settlement amounts from regression model
predicted_settlements = regression_model.predict(df_regression)

# Calculate expected loss
expected_loss = default_probabilities.mean() * predicted_settlements.mean()

print(f"Expected Loss: {expected_loss:.2f}")