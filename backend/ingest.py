from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingModel
import chromadb
from chromadb.utils import embedding_functions
import magic

chroma_client = chromadb.PersistentClient(path="./chroma_data")
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

def generate_positional_id(path: str, chunk_index: int) -> str:
    """Generates a simple, unique ID using the document path and its index."""
    safe_path = path.replace("/", "_").replace("\\", "_").replace(".", "_")
    return f"{safe_path}-chunk-{chunk_index:04d}"

def ingest(path: str ="test.txt"):
    """
    Ingests a local document into a persistent ChromaDB Collection
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunk_texts = []
    metadatas = []
    
    mime_type = magic.from_file(path, mime=True)
    
    if "pdf" in mime_type:
        loader = PyPDFLoader(path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        chunk_texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.metadata.get("source", path),
                "page": chunk.metadata.get("page", "N/A"),
                "doc_type": "pdf"
            }
            for chunk in chunks
        ]
    elif "text" in mime_type: 
        with open(path, "r", encoding="utf-8") as f:
            text_content = f.read()
        chunks = text_splitter.split_text(text_content)
        chunk_texts = [chunk for chunk in chunks]
        metadatas = [
            {
                "source": path,
                "chunk_index": i,
                "doc_type": "txt"
            }
            for i, _ in enumerate(chunk_texts)
        ]
    else:
        print("Unsupported file type", file=sys.stderr)
        return
            
    embeddings_response = embedding_model.get_embeddings(chunk_texts)
    embeddings_vector = [e.values for e in embeddings_response]
    
    ids = [
        generate_positional_id(path, i)
        for i, _ in enumerate(chunk_texts)
    ]

    collection = chroma_client.get_or_create_collection(name="my_documents")
    
    collection.add(
        documents=chunk_texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings_vector
    )
    print(f"Ingested {len(chunk_texts)} chunks into 'my_documents' from {path}")
