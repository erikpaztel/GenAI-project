from google.cloud import aiplatform
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingModel
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_data") 

def generate_positional_id(path: str, chunk_index: int) -> str:
    """Generates a simple, unique ID using the document path and its index."""
    # Simple cleanup of path for ID
    safe_path = path.replace("/", "_").replace("\\", "_").replace(".", "_")
    return f"{safe_path}-chunk-{chunk_index:04d}"

def ingest(path="test.txt"):
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunk_texts = []
    metadatas = []
    
    if ".pdf" in path:
        loader = PyPDFLoader(path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        
        # 📄 PDF METADATA GENERATION
        metadatas = [
            {
                "source": chunk.metadata.get("source", path), # Often the file path
                "page": chunk.metadata.get("page", "N/A"),
                "doc_type": "pdf"
            }
            for chunk in chunks
        ]
        chunk_texts = [chunk.page_content for chunk in chunks]
        
    else: # Assumes simple text file
        with open(path, "r", encoding="utf-8") as f:
            text_content = f.read()
        chunks = text_splitter.split_text(text_content)
        chunk_texts = [chunk for chunk in chunks]
        
        # 📝 TEXT FILE METADATA GENERATION
        metadatas = [
            {
                "source": path,
                "chunk_index": i, # Track position within the file
                "doc_type": "txt"
            }
            for i, _ in enumerate(chunk_texts)
        ]
            
    embeddings_response = embedding_model.get_embeddings(chunk_texts)
    embeddings_vector = [e.values for e in embeddings_response]
    
    # Generate IDs using the positional logic
    ids = [
        generate_positional_id(path, i)
        for i, _ in enumerate(chunk_texts)
    ]

    collection = chroma_client.get_or_create_collection(name="my_documents")
    
    # 🔑 ADDING METADATA TO THE COLLECTION
    collection.add(
        documents=chunk_texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings_vector
    )
    print(f"Ingested {len(chunk_texts)} chunks into 'my_documents' from {path}")
