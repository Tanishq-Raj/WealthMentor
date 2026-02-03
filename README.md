# Financial Research Report Generator

This project is a **Retrieval-Augmented Generation (RAG)** application designed for equity research. It uses **Google Gemini** (via `langchain-google-genai`) to generate comprehensive financial reports by synthesizing information from a local knowledge base of financial documents.

## 🚀 Features

*   **Local Document RAG**: Indexes PDF and TXT files (Annual Reports, Buffett Letters, Books) using FAISS.
*   **Gemini Powered**: Uses Google's Gemini Pro/Flash models for high-quality analysis.
*   **Streamlit UI**: User-friendly interface for generating reports and managing the vector index.
*   **Cited Sources**: Explicitly references sources from the provided Internal Documents.

## 🛠️ Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   A Google Cloud API Key (with access to Gemini API)

### 2. Installation
It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (use these specific compatible versions)
pip install streamlit langchain langchain-community langchain-google-genai==2.0.5 google-generativeai==0.8.3 faiss-cpu sentence-transformers tqdm pypdf
```

### 3. File Structure
Ensure your project directory is structured as follows for the indexer to find files:

```
RAG/
├── financial_rag_app.py    # Main application
├── requirements.txt        # Dependencies
├── faiss_financial_index/  # (Generated) Vector store
└── financial_documents/    # <--- PLACE DOCUMENTS HERE
    ├── Books/              # .txt files
    ├── Financial Research Report/ # .pdf files
    └── Warren Buffet Letters/     # .pdf files
```

## ▶️ Usage

1.  **Run the application**:
    ```bash
    streamlit run financial_rag_app.py
    ```

2.  **Configure**:
    *   Open the browser URL (usually `http://localhost:8501`).
    *   Enter your **Google API Key** in the sidebar.
    *   (Optional) Click "Force Rebuild FAISS Index" if you have added new documents to `financial_documents/`.

3.  **Generate Report**:
    *   Enter a **Company Name** (e.g., "Apple Inc.").
    *   (Optional) Add a specific focus (e.g., "Capital Allocation").
    *   Click **Generate Financial Report**.

## 🔧 Troubleshooting

*   **API Key Error**: Ensure no spaces are copy-pasted. The app now auto-strips whitespace.
*   **Model Not Found**: If you see a 404 for the model, restart the app. The code has been updated to use `gemini-1.5-flash` / `gemini-1.5-pro`.
*   **FAISS Import Error**: Using `faiss-cpu` solves issues on Windows vs `faiss`.

## 📄 License
[Your License Here]

---

## 🎯 Project Vision & Deep Dive

### What is this project doing?
This project acts as an **autonomous AI Financial Analyst**. It doesn't just "chat"; it reads, synthesizes, and analyzes thousands of pages of dense financial documents (annual reports, letters, books) to produce high-quality equity research reports comparable to those from junior analysts at top firms.

### Purpose
The purpose is to **democratize institutional-grade financial insight**. Traditionally, thorough equity research requires access to expensive terminals (Bloomberg/FactSet) and hours of manual reading. This tool bridges that gap by letting anyone build a personal, AI-powered research engine using their own document library.

### Why this? (The Problem)
*   **Information Overload**: Investors are drowning in data. A single 10-K is 100+ pages. Buffett's letters span decades.
*   **Generic AI Limits**: Asking standard ChatGPT "Analyze Apple" gives generic, outdated, or hallucinated info because it doesn't have *your* specific documents or the latest reports you just downloaded.
*   **Context Window**: You can't paste 50 PDF books into a standard chat window.

### 🚀 Why This is Better Than ChatGPT?
1.  **Grounded Truth (No Hallucinations)**: Unlike standard AI, this system uses **RAG**. It *must* find the answer in your documents first. If it says "Revenue grew 5%," it cites the specific page in the PDF.
2.  **Specialized Knowledge Base**: You control the "brain." By feeding it specific books (e.g., *The Intelligent Investor*) and letters, you force the AI to analyze stocks through the lens of value investing masters, not just generic internet opinion.
3.  **Audit Trail**: Every claim is backed by a source. You can verify the data, which is critical in finance.
4.  **Privacy**: Your documents stay on your machine until specific relevant snippets are processed. You aren't uploading your entire hard drive to the cloud.

### ⚙️ How it Works (The Architecture)

Here is exactly what happens when you run the application:

#### A. Document Loading (The "Eyes")
*   **What it does:** The script scans your `financial_documents` folder for text (`.txt`) and PDF (`.pdf`) files.
*   **Tech:** It uses `pypdf` to extract text from dense financial reports and `LangChain` loaders to process them.

#### B. The "Brain" (Vector Database)
*   **The Problem:** Computers don't "understand" text directly.
*   **The Solution:** It uses a model called `sentence-transformers/all-MiniLM-L6-v2`. This turns sentences into long lists of numbers (vectors). Similar concepts (e.g., "Revenue increased" and "Sales went up") get similar numbers.
*   **Storage (FAISS):** These numbers are saved into a specific folder called `faiss_financial_index`. This is **FAISS** (Facebook AI Similarity Search), an incredibly fast engine that acts as the project's long-term memory.

#### C. The "Analyst" (Google Gemini)
*   **Connection:** The app connects to Google's servers using your API Key.
*   **Model:** It uses `gemini-1.5-flash` (fast) or `gemini-1.5-pro` (smart).
*   **The Prompt:** The system uses a highly specific prompt that strictly instructs the AI to act as an expert analyst, prioritize internal documents, and separate "Context" findings from "General Knowledge."

#### D. The Interface (Streamlit)
*   **Controls:** Manages API keys, model selection, and user queries.
*   **Caching:** Uses `@st.cache_resource` to ensure the heavy AI models don't reload every time you click a button, making the app feel snappy.
