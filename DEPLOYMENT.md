# Deployment Guide: FraudShield on Vercel

Follow these steps to deploy your Financial Fraud Detection project and make it live.

## 1. Prepare your GitHub Repository
Vercel works best when your code is in a GitHub repository.
1.  Go to [GitHub](https://github.com) and create a new **Public** repository named `financial-fraud-detection`.
2.  Open your terminal in your project folder and run:
    ```bash
    git init
    git add .
    git commit -m "Initial commit: FraudShield with Supabase Auth"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/financial-fraud-detection.git
    git push -u origin main
    ```

## 2. Deploy to Vercel
1.  Go to [Vercel.com](https://vercel.com) and log in with your GitHub account.
2.  Click **"Add New"** -> **"Project"**.
3.  Import your `financial-fraud-detection` repository.
4.  **Important:** Vercel will automatically detect that this is a static site (HTML/JS).
5.  Click **"Deploy"**.

## 3. Dealing with the Backend (FastAPI)
Since Vercel is primarily for frontend/serverless, you have two options for your Python backend:

### Option A: Deploy Backend to Render (Recommended)
1.  Go to [Render.com](https://render.com).
2.  Create a new **"Web Service"**.
3.  Connect your GitHub repo.
4.  Set the **Start Command** to: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5.  Once live, copy the Render URL (e.g., `https://fraud-api.onrender.com`).
6.  **Update index.html:** Change the `fetch` URL in `predictFraud` from `localhost:8000` to your new Render URL.

### Option B: Use Supabase Edge Functions
If you want to keep everything in one place, you can move your Python logic to Supabase Edge Functions, but this requires more advanced setup.

## 4. Environment Variables
To keep your Supabase keys secure, you should eventually use Vercel Environment Variables:
1.  In your Vercel Dashboard, go to **Settings** -> **Environment Variables**.
2.  Add `SUPABASE_URL` and `SUPABASE_KEY`.
3.  Update your code to read these (Note: For pure HTML/JS, you will need to use a build tool like Vite to hide these properly).

---
**Your site will be live at: https://financial-fraud-detection.vercel.app**
