from vertexai.language_models import TextEmbeddingModel
from google import genai
import chromadb
import os
from chromadb.utils import embedding_functions
from typing import Dict, Any, List

project_id = os.getenv('GOOGLE_PROJECT')
location_id = os.getenv('GOOGLE_LOCATION')
model_id = os.getenv('GOOGLE_MODEL')
vertexai = os.getenv('GOOGLE_GENAI_VERTEXAI')
chroma_client = chromadb.PersistentClient(path="./chroma_data")
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
llm = genai.Client(vertexai=True, project=project_id, location=location_id)

def query_and_augment(user_question: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Queries the ChromaDB collection and uses the results to generate
    an answer with an LLM. Returns a dictionary with the answer and sources.
    """
    
    try:
        collection = chroma_client.get_collection(name="my_documents")
    except chromadb.errors.CollectionNotFoundError:
        return {
            "answer": "Error: 'my_documents' collection not found. Please run your ingest() function first.",
            "sources": []
        }
    except Exception as e:
        return {
            "answer": f"Error connecting to ChromaDB: {e}",
            "sources": []
        }
    
    query_embedding_response = embedding_model.get_embeddings([user_question])
    query_vector = query_embedding_response[0].values

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    retrieved_docs = results.get('documents', [[]])[0]
    retrieved_metadatas = results.get('metadatas', [[]])[0]
    
    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant information in the documents to answer your question.",
            "sources": []
        }

    context_chunks = []
    sources_for_ui = [] # List to hold clean source info for the final return
    
    for doc, meta in zip(retrieved_docs, retrieved_metadatas):
        chunk_lines = []
        
        # --- Extract metadata for both Context and UI ---
        source = meta.get('source', 'N/A')
        page = meta.get('page', 'N/A')
        
        # Format for the LLM context
        chunk_lines.append(f"Source: {source}")
        if page != 'N/A':
            chunk_lines.append(f"Page: {page}")
        if 'chunk_index' in meta:
             chunk_lines.append(f"Chunk Index: {meta['chunk_index']}")
        chunk_lines.append("---")
        chunk_lines.append(doc)
        context_chunks.append("\n".join(chunk_lines))

        # Format for the UI return
        ui_source = f"{os.path.basename(source)}" # Use basename for cleaner display if it's a file path
        if page != 'N/A':
            ui_source += f" (Page {page})"
        sources_for_ui.append(ui_source)
    
    context_string = "\n\n====================\n\n".join(context_chunks)
    
    SYSTEM_PROMPT = (
        "You are an expert Q&A system. Answer the user's question ONLY based on "
        "the context provided below. If the context does not contain the answer, "
        "state clearly that the answer is not available in the provided documents. "
        "Do not use external knowledge."
    )
    
    augmented_prompt = (
        f"<<SYSTEM INSTRUCTION>>\n{SYSTEM_PROMPT}\n\n"
        f"<<CONTEXT>>\n{context_string}\n"
        f"<<USER QUERY>>\n{user_question}"
    )
    
    try:
        response = llm.models.generate_content(contents=augmented_prompt, model='gemini-2.5-flash')
        return {
            "answer": response.text,
            "sources": sources_for_ui
        }
    except Exception as e:
        print(f"Error generating content with LLM: {e}")
        return {
            "answer": f"Error generating content with LLM: {e}",
            "sources": []
        }
