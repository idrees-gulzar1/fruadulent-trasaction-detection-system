# PROJECT THESIS: AI-Powered Financial Fraud Detection System (FraudShield)

## 1. Executive Summary
This project implements a comprehensive Financial Fraud Detection System designed to monitor, analyze, and predict fraudulent activities in real-time. By leveraging Machine Learning (ML) architectures, specifically Random Forest, the system provides high-fidelity risk assessments based on behavioral patterns derived from the PaySim dataset. The system features a dual-mode architecture: a high-performance Python FastAPI backend for AI inference and a robust JavaScript-based local scoring engine for fail-safe redundancy.

---

## 2. Technical Architecture & Methodology

### 2.1 Front-End Layer
- **Technology Stack:** HTML5, Vanilla CSS3, JavaScript (ES6+).
- **Core Features:** Real-time risk visualization using Chart.js (Radar, Bar, Doughnut charts), Supabase integration for persistent data storage and authentication, and a responsive transaction analysis engine.

### 2.2 Backend Inference Engine (FastAPI)
- **Framework:** FastAPI (Python 3.x) chosen for its high concurrency and asynchronous capabilities.
- **Model Deployment:** The model is serialized using `joblib` and served via a REST API endpoint (`/predict`).
- **Feature Engineering:** Incoming transaction data is pre-processed into a 6-feature vector:
  1. `amount`: Transaction value.
  2. `type_index`: Categorical encoding of transaction types (PAYMENT, TRANSFER, etc.).
  3. `hour`: Temporal feature derived from the transaction timestamp.
  4. `oldbalance`: Initial sender balance.
  5. `newbalance`: Final sender balance.
  6. `destold`: Recipient's initial balance.

### 2.3 Machine Learning Model
- **Algorithm:** Random Forest Classifier.
- **Why Random Forest?** It handles non-linear relationships and high-dimensional data effectively while being less prone to overfitting compared to single decision trees.
- **Training Data:** Simulated data inspired by the Kaggle PaySim dataset, representing over 6.3 million transactions.

---

## 3. Core Working Logic

### 3.1 The Detection Pipeline
1. **Input:** The user submits transaction details (Amount, Type, Balances).
2. **Transmission:** Data is sent via AJAX/Fetch to the FastAPI backend.
3. **Feature Mapping:** The backend maps types to numerical indices (e.g., TRANSFER -> 1).
4. **Inference:** The Random Forest model outputs a probability score (0.0 to 1.0).
5. **Verdict Generation:**
   - **Score >= 50%:** Flagged as FRAUDULENT.
   - **Score < 50%:** Classified as LEGITIMATE.
6. **Heuristic Reasoning:** The system calculates "signals" (High/Medium/Low) across four domains: Amount, Timing, Balance Drain, and Pattern.

### 3.2 Dual-Mode Fallback System
The system implements a "Live API" toggle. If the Python backend is unreachable, the system automatically switches to a **Heuristic Scoring Engine** implemented in JavaScript. This ensures 100% uptime and allows for edge-computing analysis when connectivity is low.

---

## 4. Implementation Details

- **Deployment:** The project is configured for Vercel using a serverless Python runtime.
- **Database:** Supabase (PostgreSQL) is used to store prediction history for audit logging and trend analysis.
- **Security:** CORS (Cross-Origin Resource Sharing) is enabled to allow secure communication between the frontend dashboard and the API.

---

## 5. Conclusion
FraudShield demonstrates the practical application of AI in fintech. By combining real-time ML inference with interactive data visualization, it provides a powerful tool for financial institutions to detect sophisticated fraud patterns such as account takeovers and money laundering attempts (e.g., high-volume transfers to zero-balance accounts).
