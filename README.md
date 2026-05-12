# FraudShield — Full Stack Setup

This project now includes a Python backend that hosts a real Machine Learning model (Random Forest) to analyze transaction risk.

## Quick Start (Local Setup)

### 1. Install Python Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r backend/requirements.txt
```

### 2. Generate the ML Model
Before running the server, you need to generate the model file:
```bash
python backend/generate_mock_model.py
```
This will create `backend/fraud_model.pkl`.

### 3. Start the Backend API
Run the FastAPI server:
```bash
python backend/main.py
```
The server will start at `http://localhost:8000`.

### 4. Use the Dashboard
1. Open `index.html` in your browser.
2. In the navigation bar, toggle the **"Live API"** switch to **ON**.
3. Now, when you click "Analyze Transaction", the dashboard will send the data to your Python server, get the prediction from the real ML model, and display the result.

---

## Folder Structure
- `index.html`: The single-file frontend dashboard.
- `backend/`:
    - `main.py`: The FastAPI server.
    - `generate_mock_model.py`: Script to train and save a dummy ML model.
    - `fraud_model.pkl`: The saved model file (created after step 2).
    - `requirements.txt`: Python dependencies.
- `REAL_DATA_INTEGRATION.md`: Conceptual guide for connecting actual production data.

## Note on "Live API" Mode
When the "Live API" toggle is **OFF**, the website uses a built-in JavaScript simulation (local scoring). When it is **ON**, it communicates with the Python backend. If the backend is unreachable, it will automatically fall back to local mode.
