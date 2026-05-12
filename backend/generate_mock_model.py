import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# 1. Create dummy data that mimics the PaySim structure
# Columns: amount, type_index, hour, oldbalance, newbalance, destold
# We use a simple Random Forest for this example
data = {
    'amount': [100, 500000, 20, 1000000, 50, 90000],
    'type_index': [0, 1, 0, 1, 2, 1], # 0: PAYMENT, 1: TRANSFER, etc.
    'hour': [10, 2, 14, 3, 12, 1],
    'oldbalance': [200, 500000, 100, 1000000, 1000, 90000],
    'newbalance': [100, 0, 80, 0, 950, 0],
    'destold': [0, 0, 500, 0, 0, 0],
    'is_fraud': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# 2. Train a real model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 3. Save the model
joblib.dump(model, 'backend/fraud_model.pkl')
print("✅ Mock model generated and saved to backend/fraud_model.pkl")
