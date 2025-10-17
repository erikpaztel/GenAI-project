from google.cloud import aiplatform, storage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingModel
from datetime import datetime
import os
import json

bucket_uri = os.getenv('GOOGLE_BUCKET')
project_id = os.getenv('GOOGLE_PROJECT')
location_id = os.getenv('GOOGLE_LOCATION')
model_id = os.getenv('GOOGLE_MODEL')
vertexai = os.getenv('GOOGLE_GENAI_VERTEXAI')
search_endpoint = os.getenv('GOOGLE_SEARCH_ENDPOINT')

aiplatform.init(project=project_id, location=location_id)

def generate_positional_id(path: str, chunk_index: int) -> str:
    """Generates a simple, unique ID using the document path and its index."""
    # Simple cleanup of path for ID
    safe_path = path.replace("/", "_").replace("\\", "_").replace(".", "_")
    return f"{safe_path}-chunk-{chunk_index:04d}"

def generate_jsonl_blob_name(path: str) -> str:
    """Generates a unique GCS blob name for the JSONL file."""
    # Use the basename of the original file and ensure it's unique
    document_base_name = os.path.basename(path).split('.')[0]
    # Naming convention: {document_name}_{timestamp}.jsonl for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{document_base_name}_{timestamp}.json"

def create_vertex_restricts(metadata: dict) -> list:
    """Converts Python metadata into Vertex AI 'restricts' (categorical) and 'numeric_restricts' structures."""
    restricts = []
    numeric_restricts = []
    
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            # Numeric filtering requires explicit structure
            numeric_restricts.append({
                "namespace": key,
                "value_int": int(value) if isinstance(value, int) else None,
                "value_float": float(value) if isinstance(value, float) else None,
            })
        elif isinstance(value, str):
            # Categorical filtering uses the main 'restricts' list
            restricts.append({
                "namespace": key, 
                "allow": [value] 
            })
            
    # Combine the results into the final record structure (which must be combined in the JSONL record)
    return {"restricts": restricts, "numeric_restricts": numeric_restricts}

def ingest(path="test.txt", output_dir = "tmp/"):
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
        
        #PDF METADATA GENERATION
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
        
        #TEXT FILE METADATA GENERATION
        metadatas = [
            {
                "source": path,
                "chunk_index": i, # Track position within the file
                "doc_type": "txt"
            }
            for i, _ in enumerate(chunk_texts)
        ]
    
    if not chunk_texts:
        print(f"No content found or chunks generated for {path}")
        return

    embeddings_response = embedding_model.get_embeddings(chunk_texts)
    embeddings_vectors = [e.values for e in embeddings_response] # Extract the list of vectors

    document_base_name = os.path.basename(path).split('.')[0]
    output_file_path = os.path.join(output_dir, f"{document_base_name}_embeddings.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    
    records_count = 0

    with open(output_file_path, 'w', encoding='utf-8') as f:
        # Iterate over all data points simultaneously
        for i, vector in enumerate(embeddings_vectors):
            
            # Generate ID and Vertex AI 'restricts'
            chunk_id = generate_positional_id(path, i)
            vertex_restricts = create_vertex_restricts(metadatas[i])
            
            # Construct the final Vertex AI JSON record
            record = {
                "id": chunk_id,
                "embedding": vector,
                "restricts": vertex_restricts
            }
            
            # Write the JSON object as a single line (JSONL format)
            f.write(json.dumps(record) + '\n')
            records_count += 1

    print(f"\nSuccessfully processed {path}.")
    print(f"Generated {records_count} vector records in: {output_file_path}")
    
    destination_blob_name = generate_jsonl_blob_name(path)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_uri)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(output_file_path)

    print(f"File {output_file_path} uploaded to gs://{bucket_uri}/{destination_blob_name}")

def trigger_batch_update(index_name: str, gcs_folder_uri: str, overwrite: bool = False):
    """Triggers the batch update job for the Vector Search Index."""

    index = aiplatform.MatchingEngineIndex(index_name=index_name)
    
    print(f"Triggering batch update for index {index_name}...")
    
    operation = index.update_embeddings(
        contents_delta_uri=gcs_folder_uri,
        is_complete_overwrite=overwrite # False for partial update
    )
    
    # Wait for the operation to complete (can take a long time)
    operation.wait_for_resource() 
    
    print("Batch update operation finished.")
    print(f"Operation Name: {operation.operation.name}")
    print(f"Result: {operation.display_name}")
