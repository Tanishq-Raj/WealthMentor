import streamlit as st
import os
import warnings
from tqdm import tqdm

# --- Langchain and LLM Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Application Configuration ---
st.set_page_config(page_title="Financial RAG Analyst", layout="wide")
st.title("📈 Financial Research Report Generator")
st.subheader("Powered by Local Documents and Generative AI")

# --- Path Configurations (IMPORTANT: Adjust to your LOCAL paths) ---
# Path for storing/loading the FAISS index LOCALLY
FAISS_INDEX_LOCAL_DIR = "faiss_financial_index"  # Create this directory in the same folder as your app.py
# Paths to your LOCAL document folders if rebuilding the index
LOCAL_DOCS_BASE_PATH = "financial_documents"  # Create this folder and subfolders locally
BOOKS_DIR_LOCAL = os.path.join(LOCAL_DOCS_BASE_PATH, "Books")
REPORTS_DIR_LOCAL = os.path.join(LOCAL_DOCS_BASE_PATH, "Financial Research Report")
LETTERS_DIR_LOCAL = os.path.join(LOCAL_DOCS_BASE_PATH, "Warren Buffet Letters")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Helper Functions ---
def format_docs(docs):
    if not docs: return "No relevant documents found in the knowledge base for this query."
    return "\n\n".join(f"**Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Page: {doc.metadata.get('page', 'N/A')}**\n{doc.page_content}" for doc in docs)

# --- Caching Functions for Models and Data ---
@st.cache_resource
def get_embedding_model():
    st.write("Initializing embedding model (HuggingFace)... This happens once.")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

@st.cache_resource
def load_or_build_faiss_index(_embeddings_model, force_rebuild, faiss_path, books_path, reports_path, letters_path):
    vectorstore = None
    faiss_index_file = os.path.join(faiss_path, "index.faiss")
    faiss_pkl_file = os.path.join(faiss_path, "index.pkl")

    os.makedirs(faiss_path, exist_ok=True) # Ensure local FAISS directory exists

    if not force_rebuild and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
        with st.spinner(f"Loading existing FAISS index from {faiss_path}..."):
            try:
                vectorstore = FAISS.load_local(
                    faiss_path,
                    _embeddings_model,
                    allow_dangerous_deserialization=True
                )
                st.success("FAISS index loaded successfully.")
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}. Will attempt to rebuild.")
                vectorstore = None
    else:
        if force_rebuild:
            st.info("FORCE_REBUILD_INDEX is True. Rebuilding FAISS index...")
        else:
            st.info(f"FAISS index not found at {faiss_path}. Building new index...")

    if vectorstore is None:
        with st.spinner("Building new FAISS index. This may take a while... Ensure local document paths are correct."):
            all_docs = []
            # --- Document Loading (from LOCAL paths) ---
            st.write(f"Checking local document paths: Books: {books_path}, Reports: {reports_path}, Letters: {letters_path}")
            folder_paths_local = {
                "books": books_path,
                "reports": reports_path,
                "letters": letters_path
            }

            # Load TXT files
            if os.path.exists(folder_paths_local['books']) and os.path.isdir(folder_paths_local['books']):
                txt_loader = DirectoryLoader(folder_paths_local['books'], glob="**/*.txt", loader_cls=TextLoader, show_progress=False, use_multithreading=True, silent_errors=True)
                try: book_docs = txt_loader.load(); all_docs.extend(book_docs)
                except Exception as e: st.warning(f"Error loading from Books: {e}")
            else: st.warning(f"Books path not found or not a directory: {folder_paths_local['books']}")

            # Load PDF files
            pdf_folders_map_local = {"Reports": folder_paths_local['reports'], "Letters": folder_paths_local['letters']}
            for folder_name, folder_path in pdf_folders_map_local.items():
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    pdf_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.lower().endswith('.pdf')]
                    for pdf_file_path in tqdm(pdf_files, desc=f"Processing PDFs in {folder_name}"): # tqdm might not show well in Streamlit background
                        try:
                            documents = PyPDFLoader(pdf_file_path, extract_images=False).load_and_split()
                            all_docs.extend(documents)
                        except Exception as e: st.warning(f"Could not load/process {pdf_file_path}: {e}")
                else: st.warning(f"{folder_name} path not found or not a directory: {folder_path}")

            if not all_docs:
                st.error("No documents loaded. Cannot build FAISS index. Please check local paths and content.")
                return None

            st.write(f"Total documents loaded: {len(all_docs)}")
            chunked_docs = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250).split_documents(all_docs)
            st.write(f"Documents split into {len(chunked_docs)} chunks.")
            if not chunked_docs:
                st.error("No documents chunked. Cannot build FAISS index.")
                return None

            vectorstore = FAISS.from_documents(chunked_docs, _embeddings_model)
            st.success("New FAISS vector store created.")
            try:
                vectorstore.save_local(faiss_path)
                st.success(f"New FAISS index saved to {faiss_path}")
            except Exception as e:
                st.error(f"Error saving new FAISS index: {e}")
    return vectorstore

@st.cache_resource
def get_llm(model_name):
    st.write(f"Initializing LLM: {model_name}... This happens once.")
    try:
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    except Exception as e:
        st.error(f"Failed to initialize LLM '{model_name}': {e}. Check API key and model name.")
        return None

@st.cache_data # Prompt template can be cached with cache_data
def get_rag_prompt_template_string():
    return """
You are an expert financial analyst AI assistant, tasked with producing a **highly detailed, well-researched, and comprehensive** equity research report for long-term investors. Your analysis must be thorough, insightful, and demonstrate a deep understanding of both the provided context and broader financial principles.
You have access to a set of **Internal Documents (Context)** provided below. This context (Buffett Letters, broker reports, investing books) is highly relevant and **must be prioritized and meticulously analyzed.**
**Your Task:**
Given the company name or ticker: **{company_name}**
Generate a comprehensive and in-depth financial report. You should:
1.  **Exhaustively utilize and synthesize information from the "Internal Documents (Context)"**. When using information from these documents, implicitly cite or refer to its origin (e.g., "As detailed in Buffett's 1993 letter on page X...", "The Q1 broker report projects Y...", "Drawing from the 'Valuation' chapter in 'The Intelligent Investor' provided..."). Be specific where possible.
2.  **Intelligently supplement this with your general financial knowledge and understanding of the company, its industry, and the macroeconomic environment.** This external knowledge is crucial for providing a holistic view, explaining complex concepts in detail, elaborating on industry dynamics not covered in the documents, and offering a richer, more complete picture where the provided documents are limited. Aim for depth in these explanations.
3.  **Clearly differentiate between insights drawn directly from the provided context and those based on your general knowledge.** For instance: "The provided documents do not specify recent R&D expenditures; however, based on general industry trends for a company of this scale, R&D typically constitutes X% of revenue..." or "While the context highlights historical performance, broader market analysis indicates future headwinds such as..."
4.  **If the provided context directly contradicts your general knowledge for a specific point, prioritize the information from the provided context for that point,** but you may note the discrepancy and offer a brief explanation if it's significant for a nuanced understanding.
5.  **If the context is sparse or silent on a specific section of the report, explicitly state this regarding the context-derived information, and then provide a detailed analysis based on your extensive general knowledge,** citing common financial theories, market data, or typical industry practices where appropriate.
**Internal Documents (Context):**
---
{context}
---
**Report Structure:**
Please structure your report with the following sections. For each section, first provide a detailed analysis based on the **Internal Documents (Context)**, then provide a thorough and expansive analysis based on your **General Knowledge**. Ensure seamless integration and synthesis of information where possible.
1.  **Executive Summary:**
    *   *From Context:* A concise yet comprehensive summary of key findings, material facts, and overall outlook strictly derived from the provided documents.
    *   *General Knowledge:* A broader strategic overview, incorporating current market sentiment, recent significant company news (if publicly known and relevant), and a nuanced overall investment thesis. Elucidate the 'why' behind any conclusions.
2.  **Business Overview and Competitive Moat:**
    *   *From Context:* A detailed description of the business model, core products/services, revenue streams, and specific competitive advantages (its 'economic moat') as explicitly mentioned or strongly implied in the context.
    *   *General Knowledge:* Elaborate on the company's market positioning, value chain, key dependencies, and the sustainability of its moat. Discuss Porter's Five Forces if applicable. Compare with key competitors not mentioned in the context.
3.  **Historical Financial Analysis (Revenue, EPS, Margins, Cash Flow, Debt):**
    *   *From Context:* A thorough summary of historical financial performance (e.g., revenue growth, EPS trends, margin analysis, cash flow generation, debt levels) *if and as detailed in the context*. Analyze trends and key financial ratios mentioned.
    *   *General Knowledge:* If context is limited, discuss publicly known historical financial trends over multiple years. Analyze the quality of earnings, financial health, and efficiency ratios, explaining their significance.
4.  **Forecasted Earnings and Valuation Analysis:**
    *   *From Context:* Detailed breakdown of any forecasts, valuation metrics (P/E, P/S, EV/EBITDA, DCF inputs/outputs), or price targets *found specifically in the context*. Explain the assumptions behind any context-provided valuations.
    *   *General Knowledge:* Discuss a range of appropriate valuation methodologies (e.g., DCF, comparable company analysis, precedent transactions). Provide a qualitative discussion on growth drivers and their potential impact on future earnings. If no specific valuation is in context, discuss general valuation considerations for the sector.
5.  **Management Quality, Corporate Governance, and Capital Allocation:**
    *   *From Context:* In-depth insights on the management team's philosophy, track record (as per documents), alignment with shareholder interests, and capital allocation strategies (dividends, buybacks, M&A, reinvestment) explicitly mentioned or inferred from the provided letters and books.
    *   *General Knowledge:* Discuss the general reputation and tenure of key executives, board structure, and any notable corporate governance practices. Evaluate capital allocation decisions against industry best practices and long-term value creation principles.
6.  **Industry Analysis, Macroeconomic Factors, and Economic Moats:**
    *   *From Context:* Detailed analysis of relevant industry structure, trends, and competitive dynamics mentioned, and precisely how they affect the company's moat, based on the context.
    *   *General Knowledge:* Provide a comprehensive analysis of the current state and outlook for the industry, including growth drivers, disruptive technologies, regulatory landscape, and key success factors. Discuss relevant macroeconomic factors (e.g., interest rates, inflation, GDP growth) and their specific impact on the company and its moat.
7.  **Key Risk Factors and ESG Considerations:**
    *   *From Context:* Identify and thoroughly explain key risks (macro, industry, regulatory, company-specific, operational) *explicitly highlighted or strongly implied in the context*.
    *   *General Knowledge:* Elaborate on other significant risks commonly associated with the company or its sector that may not be in the documents. Discuss relevant Environmental, Social, and Governance (ESG) factors, controversies, or opportunities, and their potential financial impact.
8.  **Analyst Recommendations and Strategic Outlook (Investment Thesis):**
    *   *From Context:* Any analyst ratings (e.g., BUY, HOLD, SELL with target prices) or specific strategic outlooks/commentary *present in the context*.
    *   *General Knowledge:* Synthesize all the above into a cohesive investment thesis. Discuss potential catalysts and headwinds. Offer a balanced strategic outlook, considering various scenarios. If appropriate, you can mention general Wall Street sentiment if widely known, but clearly label it as such.
Your tone must be professional, authoritative, clear, and deeply analytical. The report should be structured for easy readability by sophisticated investors.
"""

# --- Main Streamlit App Logic ---

# Sidebar for API Key and Controls
with st.sidebar:
    st.header("Configuration")
    # API Key Input
    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_API_KEY = ""

    api_key_input = st.text_input("Google API Key for Gemini", type="password", value=st.session_state.GOOGLE_API_KEY)
    if api_key_input:
        # Strip any leading/trailing whitespace which often happens with copy-paste
        cleaned_key = api_key_input.strip()
        st.session_state.GOOGLE_API_KEY = cleaned_key
        os.environ["GOOGLE_API_KEY"] = cleaned_key # Set for Langchain/Google libraries

    # Model Selection
    st.subheader("LLM Model Selection")
    # List available models (you can pre-populate or fetch if your key is already entered)
    # For simplicity, we'll offer a few common ones. Ensure your key has access.
    available_models_for_ui = ["gemini-1.5-flash", "gemini-1.5-pro"] # Add more if needed
    
    if "llm_model_choice" not in st.session_state:
        st.session_state.llm_model_choice = available_models_for_ui[0] # Default to flash

    llm_model_choice = st.selectbox(
        "Choose LLM Model:",
        available_models_for_ui,
        index=available_models_for_ui.index(st.session_state.llm_model_choice)
    )
    st.session_state.llm_model_choice = llm_model_choice


    # Force Rebuild FAISS Index
    st.subheader("FAISS Index")
    force_rebuild = st.checkbox("Force Rebuild FAISS Index", value=False)
    if st.button("Clear FAISS Index Cache & App State"):
        # Clear specific cache entries
        st.cache_resource.clear() # Clears all @st.cache_resource
        st.cache_data.clear() # Clears all @st.cache_data
        # Reset relevant session state items if any
        for key in list(st.session_state.keys()):
            if key not in ['GOOGLE_API_KEY', 'llm_model_choice']: # Keep essential inputs
                 del st.session_state[key]
        st.rerun()


# Check if API key is provided
if not st.session_state.GOOGLE_API_KEY:
    st.warning("Please enter your Google API Key in the sidebar to proceed.")
    st.stop()

# Initialize models and vector store
embedding_model = get_embedding_model()
vectorstore = load_or_build_faiss_index(
    embedding_model,
    force_rebuild,
    FAISS_INDEX_LOCAL_DIR,
    BOOKS_DIR_LOCAL,
    REPORTS_DIR_LOCAL,
    LETTERS_DIR_LOCAL
)

if not vectorstore:
    st.error("Failed to load or build FAISS vector store. Application cannot proceed.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
llm = get_llm(st.session_state.llm_model_choice)

if not llm:
    st.error("Failed to initialize LLM. Application cannot proceed.")
    st.stop()

# --- RAG Chain Setup ---
rag_prompt_template_str = get_rag_prompt_template_string()
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template_str)

rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x["retrieval_query"])), "company_name": lambda x: x["company_name"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)
st.success("RAG system ready!")

# --- User Inputs for Report Generation ---
st.header("Generate New Report")
company_name = st.text_input("Enter Company Name or Ticker:", placeholder="e.g., Apple Inc. or AAPL")
specific_focus = st.text_area("Specific Focus for Retrieval (Optional):", placeholder="e.g., competitive advantages, capital allocation, and future growth strategy")

if st.button("Generate Financial Report", type="primary"):
    if not company_name:
        st.warning("Please enter a company name or ticker.")
    else:
        with st.spinner(f"Generating report for {company_name}... This may take a few moments."):
            # Construct retrieval query
            if specific_focus:
                retrieval_q = f"Financial analysis, competitive moat, management, risks, and specific focus on '{specific_focus}' for {company_name}"
            else:
                retrieval_q = f"Comprehensive analysis of {company_name}, including its business, financials, valuation, management, risks, industry, and ESG factors."

            st.write(f"🧠 Using retrieval query: \"{retrieval_q}\"")

            # For transparency, show retrieved sources (optional, can be verbose)
            # retrieved_docs_for_display = retriever.invoke(retrieval_q)
            # if retrieved_docs_for_display:
            #     st.subheader("Retrieved Context Snippets (Top Sources):")
            #     seen_sources_display = set()
            #     for i, doc_disp in enumerate(retrieved_docs_for_display[:3]): # Show top 3
            #         source_file_disp = os.path.basename(doc_disp.metadata.get('source', 'Unknown'))
            #         if source_file_disp not in seen_sources_display:
            #             st.caption(f"Source: {source_file_disp} | Page: {doc_disp.metadata.get('page', 'N/A')}")
            #             # st.markdown(f"> {doc_disp.page_content[:300]}...") # Display snippet
            #             seen_sources_display.add(source_file_disp)
            # else:
            #     st.write("No specific documents retrieved for this query focus from the local DB.")

            inputs = {"company_name": company_name, "retrieval_query": retrieval_q}
            try:
                response = rag_chain.invoke(inputs)
                st.subheader(f"Generated Financial Report for {company_name}")
                st.markdown(response) # Use markdown to render formatted LLM output
            except Exception as e:
                st.error(f"An error occurred during LLM generation: {e}")
                # More detailed error for debugging if needed
                # st.exception(e)

st.markdown("---")
st.caption("Ensure your local document paths and FAISS index path are correctly configured at the top of the script.")
st.caption("If rebuilding the index, make sure your document folders (Books, Reports, Letters) are populated locally.")