# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from preprocess import preprocess_data

# Load the data
df = pd.read_csv('Data/test_Y3wMUE5_7gLdaTN.csv')

# Derive Loan_Status from LoanAmount: Approved if LoanAmount > 0, Denied if LoanAmount = 0
df['Loan_Status'] = df['LoanAmount'].apply(lambda x: 1 if x > 0 else 0)

# Preprocess the data using the preprocess_data function
df = preprocess_data(df)

# Features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)  # Drop Loan_ID, which is not used for prediction
y = df['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'loan_model.pkl')
print("Model trained and saved as loan_model.pkl")

