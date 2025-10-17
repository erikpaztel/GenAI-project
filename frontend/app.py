import streamlit as st
import time
import os

# --- Import Backend Logic ---
# Make sure the 'backend' folder is in the same root directory as 'frontend'
from backend.ingest import ingest

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📄",
    layout="wide"
)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything about it. 📄"}]
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# --- Sidebar for Document Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )

    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a moment."):
                # 1. Define the path for the 'uploaded' directory
                upload_dir = "uploaded"
                
                # 2. Check if the directory exists, create it if it doesn't
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                
                # 3. Save the uploaded file to the directory
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 4. Call the ingestion logic from the backend
                ingest(file_path)
                
                st.success("Document processed and ready for questions!")
                st.session_state.document_processed = True
    else:
        st.session_state.document_processed = False
        st.info("Please upload a file to begin.")


# --- Main Chat Interface ---
st.title("RAG Document Q&A System")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(source)

# Accept user input
prompt = st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.document_processed
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Mock response for demonstration
        assistant_response = "This is a mock answer based on the document's content."
        
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

        with st.expander("View Sources"):
            st.write("Source 1: 'The first relevant chunk of text from the document...'")
            st.write("Source 2: 'Another relevant paragraph that contributed to the answer...'")
        
        mock_sources = [
            "Source 1: 'The first relevant chunk of text from the document...'",
            "Source 2: 'Another relevant paragraph that contributed to the answer...'"
        ]

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": mock_sources
    })
