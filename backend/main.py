from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time

app = FastAPI(title="FraudShield AI API")

# Enable CORS so the index.html can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
if not os.path.exists(MODEL_PATH):
    # This is a fallback in case the user hasn't run the generator yet
    model = None
else:
    model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    amount: float
    type: str
    hour: int
    oldbalance: float
    newbalance: float
    destold: float
    model_name: str

TYPE_MAP = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4,
    "CREDIT_CARD": 5,
    "RAZORPAY": 6,
    "PAYPAL": 7
}

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict_fraud(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Run generate_mock_model.py first.")

    start_time = time.time()

    # Preprocess the data to match the training format
    input_data = pd.DataFrame([{
        "amount": tx.amount,
        "type_index": TYPE_MAP.get(tx.type, 0),
        "hour": tx.hour,
        "oldbalance": tx.oldbalance,
        "newbalance": tx.newbalance,
        "destold": tx.destold
    }])

    # Get probability from the real model
    prob = model.predict_proba(input_data)[0][1] # Probability of class 1 (Fraud)
    score = int(prob * 100)

    # Generate reasoning (In a real app, this would be from SHAP or LIME)
    reasons = []
    if tx.amount > 200000: reasons.append("extremely high volume")
    if tx.oldbalance > 0 and (tx.oldbalance - tx.newbalance) / tx.oldbalance > 0.9: reasons.append("account drainage")
    if tx.hour < 5: reasons.append("irregular timing")

    reasoning = f"AI Analysis: {score}% risk detected. " + (", ".join(reasons) if reasons else "Pattern matches standard activity.")

    process_time = round(time.time() - start_time, 4)

    return {
        "fraud_probability": score,
        "verdict": "FRAUDULENT" if score >= 50 else "LEGITIMATE",
        "confidence": "HIGH" if (score > 80 or score < 20) else "MEDIUM",
        "signals": {
            "amount": "HIGH" if tx.amount > 100000 else "MEDIUM" if tx.amount > 20000 else "LOW",
            "timing": "HIGH" if tx.hour <= 5 else "LOW",
            "balance": "HIGH" if tx.oldbalance > 0 and (tx.oldbalance - tx.newbalance) / tx.oldbalance > 0.8 else "LOW",
            "pattern": "HIGH" if tx.type == "TRANSFER" and tx.destold == 0 else "MEDIUM"
        },
        "reasoning": reasoning,
        "model": f"{tx.model_name} (Remote)",
        "time": str(process_time)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
