from google import genai
from google.cloud import aiplatform, storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingModel

project_id = os.getenv('GOOGLE_PROJECT')
location_id = os.getenv('GOOGLE_LOCATION')
model_id = os.getenv('GOOGLE_MODEL')
vertexai = os.getenv('GOOGLE_GENAI_VERTEXAI')
endpoint_id = os.getenv('GOOGLE_SEARCH_ENDPOINT')
deploy_id = os.getenv('GOOGLE_SEARCH_ENDPOINT_DEPLOY')

def retrieve_relevant_chunks(query_vector: list, endpoint_id: str, deployed_id: str, k: int = 5) -> list:
    """
    Queries the deployed Vector Search Index Endpoint with a vector and retrieves
    the top K closest document chunks (as text/metadata).
    """
    
    # 1. Get the Deployed Endpoint (Resource to Query)
    # Note: Use the MatchingEngineIndexEndpoint class/resource name
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
    
    # 2. Execute the Vector Search
    # The response contains the index ID, the neighbors (chunks), and the distance.
    results = index_endpoint.find_neighbors(
        deployed_index_id=deployed_id, # ID used during deployment (e.g., 'deployed_index_v1')
        queries=[query_vector],
        num_neighbors=k
    )
    
    # 3. Extract the original text/metadata from the results
    # The actual chunk data is typically embedded in the datapoint_id or metadata field.
    retrieved_context = []
    
    # The results structure is nested: [0] refers to the single query you sent
    for neighbor in results[0].neighbors:
        # The original ID is the datapoint_id. 
        # The text is NOT returned directly and must be retrieved separately 
        # from an external store (like BigQuery or the original file system) 
        # using the datapoint_id/chunk_id as a key.
        
        # For simplicity, if you stored the full chunk text in the datapoint_id (not recommended) 
        # or in 'restricts' (better, see below), you'd pull it here.
        chunk_id = neighbor.datapoint.datapoint_id
        
        # Best Practice: Retrieve the chunk text from the ID/metadata
        # Since the datapoint_id is your unique chunk ID, 
        # you need a separate lookup function to get the raw text.
        
        # For now, we'll return the ID and distance
        retrieved_context.append({
            "chunk_id": chunk_id,
            "distance": neighbor.distance,
            "metadata_restricts": neighbor.datapoint.restricts # Filtering metadata
        })
        
    return retrieved_context

def generate_augmented_prompt(retrieved_chunks: list, user_query: str) -> str:
    """
    Combines the retrieved chunks and the user query into a single, structured prompt.
    """
    SYSTEM_PROMPT = (
        "You are an expert Q&A system. Answer the user's question ONLY based on "
        "the context provided below. If the context does not contain the answer, "
        "state clearly that the answer is not available in the provided documents. "
        "Do not use external knowledge."
    )
    
    context_text = ""
    for chunk in retrieved_chunks:
        # NOTE: You MUST replace this with a lookup to get the actual text content
        # For this example, we assume you get the text by looking up the chunk_id
        
        # Placeholder function - replace with actual text lookup
        chunk_text = lookup_chunk_text(chunk['chunk_id']) 
        
        # Extract metadata for citation
        source_doc = next((r['allow'][0] for r in chunk['metadata_restricts'] if r['namespace'] == 'source'), 'N/A')
        page_num = next((r['allow'][0] for r in chunk['metadata_restricts'] if r['namespace'] == 'page'), 'N/A')
        
        # Format the context block with clear citations (Crucial for RAG)
        context_text += f"[SOURCE: {source_doc}, page {page_num}]\n{chunk_text}\n\n"
        
    augmented_prompt = (
        f"<<SYSTEM INSTRUCTION>>\n{SYSTEM_PROMPT}\n\n"
        f"<<CONTEXT>>\n{context_text}\n"
        f"<<USER QUERY>>\n{user_query}"
    )
    return augmented_prompt

def process_query(user_query: str) -> str:
    # 1. Initialize Embedding Model
    # Must use the SAME model used for indexing
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    
    # --- STEP 1: Embed the User Query ---
    # Convert the user's text query into a vector
    query_embedding_response = embedding_model.get_embeddings([user_query])
    query_vector = query_embedding_response[0].values
    
    # 2. Get LLM Model for Generation
    # The client instantiation can be moved outside the function if performance is critical
    client = genai.Client(
        vertexai=True, 
        project=project_id, 
        location=location_id, 
        # http_options=types.HttpOptions(api_version='v1') # Not strictly necessary unless troubleshooting/explicit versioning
    )
    
    retrieved_chunks_data = retrieve_relevant_chunks(
        query_vector=query_vector, 
        endpoint_id=endpoint_id, 
        deployed_id=deploy_id, 
        k=5
    )
    
    augmented_prompt = generate_augmented_prompt(
        retrieved_chunks_data,
        user_query=user_query
    )
    
    print("Sending augmented prompt to LLM...")
    response = client.models.generate_content(
        model=llm_model_name,
        contents=[augmented_prompt]
    )
    
    return response.text
