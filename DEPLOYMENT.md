# How to Deploy WealthMentor

The easiest way to deploy this application for free is using **Streamlit Community Cloud**.

## 1. Prepare your Repository (Important)

We need to make sure your financial documents are uploaded so the cloud app can read them.

1.  **Modify .gitignore**: I have already updated your `.gitignore` to *allow* uploading the `financial_documents` folder.
2.  **Run these commands** in your terminal to push the documents to GitHub:
    ```bash
    git add .
    git commit -m "Add financial documents for deployment"
    git push
    ```

> **Note on `faiss_financial_index`**: We do *not* upload this folder. The app will automatically rebuild it on the cloud server the first time it runs. This ensures the index is compatible with the cloud's Linux environment (yours is Windows).

## 2. Deploy on Streamlit Cloud

1.  Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign up/login with your GitHub account.
2.  Click **"New app"**.
3.  Select your repository: `Tanishq-Raj/WealthMentor`.
4.  **Configuration**:
    *   **Main file path**: `financial_rag_app.py`
    *   **Python version**: 3.10 (or 3.11/3.12)
5.  **🚨 Advanced Settings (Crucial)**:
    *   Click "Advanced settings" (or "Variables" / "Secrets").
    *   You must add your API Key here so the cloud app can use Gemini.
    *   Copy the following format:
        ```toml
        GOOGLE_API_KEY = "your-actual-google-api-key-here"
        ```
6.  Click **Deploy!**

## 3. What to Expect
*   The first time the app starts, it will take a few minutes (maybe 2-5 mins) to **Install Dependencies** and **Build the FAISS Index** from your uploaded documents.
*   Once running, anyone with the link can use your AI Analyst!

## Alternative: Render / Railway
If Streamlit Cloud fails due to size limits (your documents are ~160MB + overhead), you can try **Render.com** (Web Service, Python) but it requires more setup (Procfile, build commands). Streamlit Cloud is the recommended first choice.
