import streamlit as st
import time
import os

def display(ingest, query):
    """Main function to run the Streamlit app."""
    
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
                    upload_dir = "uploaded"
                    
                    if not os.path.exists(upload_dir):
                        os.makedirs(upload_dir)
                    
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    ingest(file_path)
                    
                    st.success("Document processed and ready for questions!")
                    st.session_state.document_processed = True
        else:
            st.session_state.document_processed = False
            st.info("Please upload a file to begin.")

    st.title("RAG Document Q&A System")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.write(source)

    prompt = st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.document_processed
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = query(prompt)
            
            for chunk in assistant_response.answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

            with st.expander("View Sources"):
                for source in assistant_response.sources:
                    st.write(source)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": assistant_response.sources
        })
