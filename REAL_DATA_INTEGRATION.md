# Integrating Real ML Models with FraudShield

To move from the current **simulated** logic to **real-world** fraud detection, follow this simplified 3-step guide.

## 1. Create the AI Backend (Python)
Since browsers can't run complex ML models (like XGBoost) efficiently, you need a small Python server.

### Install Requirements
```bash
pip install fastapi uvicorn xgboost pandas
```

### The API Code (`main.py`)
This script loads your trained model and waits for data from the website.
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
import pandas as pd

app = FastAPI()

# Allow your website to talk to this API
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load your trained model file
model = xgb.Booster()
model.load_model("fraud_model.json")

@app.post("/predict")
async def predict(data: dict):
    # Convert incoming JSON to a Format the model understands
    df = pd.DataFrame([data])
    
    # Get probability
    dmatrix = xgb.DMatrix(df)
    probability = model.predict(dmatrix)[0]
    
    return {
        "fraud_probability": int(probability * 100),
        "verdict": "FRAUDULENT" if probability > 0.5 else "LEGITIMATE"
    }
```

---

## 2. Connect the Website to the API
In your `index.html` file, find the `predictFraud` function and replace it with a `fetch` call to your new Python server.

### New Frontend Code
```javascript
async function predictFraud(tx) {
    // 1. Send data to your Python server
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tx)
    });

    // 2. Get the real AI result back
    const aiResult = await response.json();

    // 3. Return the data to update the UI
    return {
        ...aiResult,
        confidence: aiResult.fraud_probability > 80 ? "HIGH" : "MEDIUM",
        time: "0.02s" 
    };
}
```

---

## 3. Feeding Real Data
To analyze actual live transactions:

1.  **Direct Integration:** If you have a checkout page, call the `predictFraud` function right before the payment is processed.
2.  **Database Sync:** Set up a "Listener" on your database. Every time a new transaction row is added, it triggers a call to your FraudShield API.
3.  **Third-Party Webhooks:** If using Stripe or PayPal, configure their "Webhooks" to send transaction data directly to your Python API endpoint.

## Summary Checklist
- [ ] **Train Model:** Use a Python notebook to train on the [PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).
- [ ] **Save Model:** Export it as `fraud_model.json`.
- [ ] **Host API:** Deploy your Python code to a service like **Render**, **Railway**, or **AWS**.
- [ ] **Update URL:** Change `localhost:8000` in your JS to your live API link.
