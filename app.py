"""
============================================================
 MODULE 6: The Final Product — Streamlit UI (app.py)
============================================================

 WHAT WE'RE DOING:
    Turning everything we've built into a real, usable web application!
    
    Features:
      📤 Upload any PDF using a drag-and-drop file uploader
       Chat with the PDF using a ChatGPT-style interface
       Remembers chat history within the session
      🛡️ Hallucination protection built-in

 WHAT IS STREAMLIT?
    Streamlit is a Python framework that turns scripts into web apps
    with ZERO HTML/CSS/JavaScript. You just write Python!

⚠️ THE PROBLEM (The Refresh Loop):
    In Streamlit, every time you click a button or type something,
    the ENTIRE script runs again from TOP to BOTTOM.
    
    This means:
      - Your chat history? GONE.
      - Your uploaded file? Needs re-processing.
      - Your vector store? Recreated every time.
    
    It's like having amnesia after every click! 

 THE SOLUTION (Session State):
    `st.session_state` is like a "memory box" that survives reruns.
    We store our chat history, vector store, and processing status
    in session state, so nothing is lost between interactions.

▶️ HOW TO RUN:
    streamlit run app.py
"""

# ============================================================
# STEP 1: Imports
# ============================================================

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

# LLM
from langchain_groq import ChatGroq

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# PDF Loading & Chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# RAG Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load API keys
load_dotenv()


# ============================================================
# STEP 2: Page Configuration
# ============================================================

# This MUST be the first Streamlit command in the script
st.set_page_config(
    page_title=" PDF Q&A Chatbot",
    page_icon="",
    layout="wide"
)


# ============================================================
# STEP 3: Initialize Session State
# ============================================================

#  THIS IS THE SOLUTION TO THE REFRESH LOOP!
#
# st.session_state is a dictionary that persists across reruns.
# We use it to store:
#   - chat_history: all messages in the conversation
#   - vector_store: the ChromaDB instance (so we don't rebuild it)
#   - pdf_processed: flag to know if a PDF has been processed

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ============================================================
# STEP 4: Initialize the LLM & Embeddings (cached)
# ============================================================

# @st.cache_resource tells Streamlit to create these ONCE
# and reuse them across reruns. Without this, we'd create
# new instances every time the user types something!

@st.cache_resource
def get_llm():
    """Create the LLM instance (only once)."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

@st.cache_resource
def get_embeddings():
    """Load the embedding model (only once, ~90MB download on first run)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = get_llm()
embeddings = get_embeddings()


# ============================================================
# STEP 5: The RAG Prompt (with Hallucination Guard)
# ============================================================

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions based ONLY 
on the provided context from a PDF document.

RULES:
- Answer using ONLY the information in the context below
- Be concise and direct
- Use bullet points when listing multiple items

IMPORTANT: If the answer is NOT in the provided context, strictly say:
"I don't know based on the provided document."
Do NOT make up or assume any information.

CONTEXT:
{context}"""
    ),
    (
        "human",
        "{input}"
    )
])


# ============================================================
# STEP 6: PDF Processing Function
# ============================================================

def process_pdf(uploaded_file):
    """
    Takes an uploaded PDF file and:
      1. Saves it to a temp file (PyPDFLoader needs a file path)
      2. Loads and chunks the text
      3. Creates embeddings and stores in ChromaDB
      4. Returns the vector store
    """
    with st.spinner(" Reading PDF..."):
        # Streamlit gives us bytes, but PyPDFLoader needs a file path
        # So we save it to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

    with st.spinner("✂️ Splitting into chunks..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)

    with st.spinner(f" Creating embeddings for {len(chunks)} chunks..."):
        # Use a unique directory for each uploaded file
        persist_dir = os.path.join(tempfile.gettempdir(), "chroma_streamlit")
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="streamlit_pdf"
        )

    # Clean up the temp PDF file
    os.unlink(tmp_path)

    return vector_store, len(pages), len(chunks)


# ============================================================
# STEP 7: Build the UI
# ============================================================

# --- Header ---
st.title(" PDF Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it! Powered by Groq + LangChain + ChromaDB")

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your PDF here",
        type=["pdf"],
        help="Upload a PDF file to start asking questions about it"
    )

    if uploaded_file and not st.session_state.pdf_processed:
        # Process the PDF when uploaded
        vector_store, num_pages, num_chunks = process_pdf(uploaded_file)
        
        # Save to session state (survives reruns!)
        st.session_state.vector_store = vector_store
        st.session_state.pdf_processed = True
        st.session_state.chat_history = []  # clear old chats

        st.success(f" Processed: {num_pages} pages → {num_chunks} chunks")

    if st.session_state.pdf_processed:
        st.info("📋 PDF loaded and ready for questions!")
        
        # Button to reset and upload a new PDF
        if st.button("🔄 Upload New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.rerun()

    # --- About Section ---
    st.divider()
    st.markdown("""
    ### ℹ️ How it works
    1. **Upload** a PDF document
    2. The app **splits** it into small chunks
    3. Chunks are converted to **embeddings**
    4. Ask a question → app **searches** for relevant chunks
    5. Relevant chunks + question → **LLM** → Answer!
    
    _This is called **RAG** (Retrieval Augmented Generation)_
    """)


# ============================================================
# STEP 8: Display Chat History
# ============================================================

# Loop through all previous messages and display them
# This is how we "restore" the chat after a Streamlit rerun

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# STEP 9: Handle User Input
# ============================================================

# st.chat_input creates a text input fixed at the bottom of the page
user_question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=not st.session_state.pdf_processed
)

if user_question:
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Save user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    # Build and run the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Create the RAG chain
            document_chain = create_stuff_documents_chain(llm, rag_prompt)
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            rag_chain = create_retrieval_chain(retriever, document_chain)

            # Get the answer
            response = rag_chain.invoke({"input": user_question})
            answer = response["answer"]
        
        # Display the answer
        st.markdown(answer)
        
        # Show which pages were referenced (expandable)
        with st.expander(" View Source Chunks"):
            for i, doc in enumerate(response["context"], 1):
                page_num = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** (Page {page_num}):")
                st.caption(doc.page_content[:300] + "...")
                st.divider()

    # Save assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# ============================================================
# STEP 10: Empty State (when no PDF is uploaded)
# ============================================================

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF in the sidebar to get started!")
    
    # Show a helpful guide
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Drag a PDF into the sidebar uploader")
    with col2:
        st.markdown("### 2️⃣ Wait")
        st.markdown("The app processes your document automatically")
    with col3:
        st.markdown("### 3️⃣ Ask")
        st.markdown("Type any question about your document!")
