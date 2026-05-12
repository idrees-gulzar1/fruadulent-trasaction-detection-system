# EXAM CHEAT SHEET: Financial Fraud Detection (FraudShield)

## 1. Quick Technical Summary
- **Primary Goal:** Real-time prediction of fraudulent financial transactions.
- **Algorithm:** Random Forest (Ensemble method using 100+ decision trees).
- **Backend:** FastAPI (Python) for ML; Vercel for hosting.
- **Frontend:** HTML/CSS/JS with Chart.js for data visualization.
- **Database:** Supabase (PostgreSQL) for transaction history.

## 2. The Data Flow (A-Z)
1. **User Action:** User inputs transaction data on `index.html`.
2. **Request:** JS `fetch()` sends a JSON POST request to `/api/predict`.
3. **Backend Processing:**
   - `main.py` receives the JSON.
   - Categorical data (`type`) is converted to numbers using `TYPE_MAP`.
   - Data is loaded into a Pandas DataFrame.
4. **ML Inference:** `model.predict_proba()` calculates the risk percentage.
5. **Heuristic Check:** Backend checks for "Red Flags" (e.g., amount > 200k, account drainage > 90%).
6. **Response:** Backend returns a JSON object with `score`, `verdict`, and `reasoning`.
7. **UI Update:** `index.html` receives the JSON, updates charts, and shows the result in the sidebar.

## 3. Top 5 "Examiner Questions" & Answers
- **Q1: Why use Random Forest instead of a simple Decision Tree?**
  - **A:** Random Forest reduces variance and prevents overfitting by averaging multiple trees. It is more robust to noise in financial data.
- **Q2: What are the most important features for detection?**
  - **A:** `Amount` (high values), `Balance Delta` (sender losing all money), and `Transaction Type` (TRANSFER and CASH_OUT are high-risk).
- **Q3: How does the system handle backend failure?**
  - **A:** It uses a **Client-Side Fallback**. If the API is offline, a JavaScript function uses weighted heuristics to estimate risk locally.
- **Q4: How do you handle imbalanced data?**
  - **A:** (Standard Answer) By using metrics like Precision-Recall and F1-Score instead of just Accuracy, and techniques like SMOTE or class weighting during training.
- **Q5: Is the model real-time?**
  - **A:** Yes. The FastAPI backend responds in < 50ms, making it suitable for point-of-sale or online banking integration.

## 4. Key Metrics to Mention
- **Dataset Size:** 6.3 Million transactions (PaySim).
- **Inference Time:** ~0.02s - 0.05s.
- **Threshold:** 50% (Score >= 50 is Fraud).
- **Risk Signals:** Amount, Timing, Balance, Pattern.
