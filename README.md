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

## 🎯 Project Vision & Detailed Breakdown

### What is this project doing?
This project acts as an **autonomous AI Financial Analyst**. It doesn't just "chat"; it reads, synthesizes, and analyzes thousands of pages of dense financial documents (annual reports, letters, books) to produce high-quality equity research reports comparable to those from junior analysts at top firms.

### Purpose
The purpose is to **democratize institutional-grade financial insight**. Traditionally, thorough equity research requires access to expensive terminals (Bloomberg/FactSet) and hours of manual reading. This tool bridges that gap by letting anyone build a personal, AI-powered research engine using their own document library.

### Why this? (The Problem)
*   **Information Overload**: Investors are drowning in data. A single 10-K is 100+ pages. Buffett's letters span decades.
*   **Generic AI Limits**: Asking standard ChatGPT "Analyze Apple" gives generic, outdated, or hallucinated info because it doesn't have *your* specific documents or the latest reports you just downloaded.
*   **Context Window**: You can't paste 50 PDF books into a standard chat window.

### 🛠️ Tech Stack & Tools Used
The project leverages a modern AI stack specializing in natural language processing and document retrieval:

| Component | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **User Interface** | Streamlit | Provides the web-based dashboard for entering API keys, uploading/indexing files, and viewing reports. |
| **Orchestration** | LangChain | The "glue" that connects the document loaders, the vector store, and the LLM into a single pipeline. |
| **Vector Database** | FAISS | A high-performance library for searching through millions of document snippets in milliseconds. |
| **PDF Processing** | PyPDF | Extracts raw text from dense, multi-page financial PDFs. |
| **Embedding Model** | HuggingFace (`all-MiniLM-L6-v2`) | Converts text into 384-dimensional vectors that represent the meaning of the words. |
| **LLM (The Brain)** | Google Gemini (1.5 Pro/Flash) | The state-of-the-art model that reads the retrieved context and writes the final report. |

### 🧠 Machine Learning Models
The project uses two distinct types of ML models:

*   **Embedding Model (`sentence-transformers/all-MiniLM-L6-v2`)**:
    *   **Function**: This is an encoder model. It doesn't generate text; instead, it turns sentences into lists of numbers (vectors).
    *   **Why it's used**: It allows the system to perform "semantic search." If you search for "profitability," the model knows that "net income" and "margins" are related concepts, even if the exact word isn't used.
*   **Large Language Model (Google Gemini)**:
    *   **Gemini 1.5 Flash**: Used for high-speed, cost-effective generation.
    *   **Gemini 1.5 Pro**: Used for deep reasoning and complex financial analysis (better for longer reports).
    *   **Role**: It acts as the "writer," following a 300-word system prompt that instructs it to ignore generic AI "fluff" and stick to hard financial data.

### ⚙️ Entire Working Process
The application operates in four main phases:

#### Phase A: Document Ingestion (The "Learning" Phase)
1.  The app scans the `financial_documents/` folders.
2.  **Chunking**: It breaks long documents (like a 200-page PDF) into smaller pieces of 1,500 characters each (with a 250-character overlap to preserve context).
3.  **Vectorization**: Each chunk is passed through the Embedding Model to create a vector.
4.  **Indexing**: These vectors are saved into the `faiss_financial_index` folder on your local drive.

#### Phase B: Retrieval (The "Research" Phase)
1.  When you enter a company name (e.g., "Apple Inc."), the app generates a specialized retrieval query.
2.  It converts this query into a vector and asks FAISS: "Find me the 7 most relevant snippets from my library regarding this query."
3.  The system pulls these 7 chunks (the "Context").

#### Phase C: Augmented Generation (The "Writing" Phase)
1.  **Prompt Construction**: The app builds a massive prompt containing:
    *   The 7 retrieved document snippets (Grounded Truth).
    *   A structured template (8 sections: Moat, Financials, Management, etc.).
    *   Strict logic instructions: "Use the provided context first; if info is missing, use your general knowledge but label it clearly."
2.  **LLM Processing**: Gemini reads the prompt and synthesizes the final report.

#### Phase D: Output
1.  The report is rendered in Markdown format, allowing for bold headers, tables, and bullet points.
2.  The report includes **citations** (e.g., "Source: 2023_Annual_Report.pdf | Page: 42"), making the AI's claims verifiable and "hallucination-resistant."

### 🌟 Unique Highlights
*   **Hybrid Knowledge**: Unlike ChatGPT, which only knows what it was trained on, this tool combines General AI knowledge with **Your Private Documents**.
*   **Audit Trail**: Every financial claim can be traced back to a specific page in a specific document.
*   **Local Processing**: The heavy lifting of document indexing happens on your machine, ensuring your library stays private.
