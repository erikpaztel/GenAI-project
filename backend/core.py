from vertexai.language_models import TextEmbeddingModel
from google import genai
import chromadb
import os
from chromadb.utils import embedding_functions

project_id = os.getenv('GOOGLE_PROJECT')
location_id = os.getenv('GOOGLE_LOCATION')
model_id = os.getenv('GOOGLE_MODEL')
vertexai = os.getenv('GOOGLE_GENAI_VERTEXAI')
chroma_client = chromadb.PersistentClient(path="./chroma_data")
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
llm = genai.Client(vertexai=True, project=project_id, location=location_id)

def query_and_augment(user_question: str, n_results: int = 3) -> str:
    """
    Queries the ChromaDB collection and uses the results to generate
    an answer with an LLM.
    """
    
    try:
        collection = chroma_client.get_collection(name="my_documents")
    except chromadb.errors.CollectionNotFoundError:
        print("Error: 'my_documents' collection not found.")
        print("Please run your ingest() function first.")
        return "Error: Collection not found."
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return f"Error: {e}"
    
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
        return "I couldn't find any relevant information in the documents to answer your question."

    context_chunks = []
    
    for doc, meta in zip(retrieved_docs, retrieved_metadatas):
        chunk_lines = []
        
        source = meta.get('source', 'N/A')
        chunk_lines.append(f"Source: {source}")
        
        if 'page' in meta and meta['page'] != 'N/A':
            chunk_lines.append(f"Page: {meta['page']}")
        
        if 'chunk_index' in meta:
             chunk_lines.append(f"Chunk Index: {meta['chunk_index']}")

        chunk_lines.append("---")
        chunk_lines.append(doc)
        
        context_chunks.append("\n".join(chunk_lines))
    
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
        return response.text
    except Exception as e:
        print(f"Error generating content with LLM: {e}")
        return f"Error: {e}"
