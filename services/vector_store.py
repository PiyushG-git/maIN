# import os
# from pinecone import Pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from config.settings import settings

# # Initialize Pinecone
# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# index = pc.Index(settings.PINECONE_INDEX_NAME)

# # Initialize HuggingFace embedding model
# embedding_model = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#     temperature=0.0,
#     top_k=1,
#     top_p=1.0,
#     do_sample=False,
#     repetition_penalty=1.0,
# )
# model = ChatHuggingFace(llm=llm)

# def split_text(text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.split_text(text)

# async def embed_and_upsert(chunks: list[str], namespace: str):
#     print(f"Embedding and upserting {len(chunks)} chunks into namespace: {namespace}")
#     try:
#         batch_size = 100
#         total_batches = (len(chunks) + batch_size - 1) // batch_size
#         total_inserted = 0

#         print(f"🧮 Total batches to process: {total_batches} (batch size = {batch_size})")

#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i:i + batch_size]
#             current_batch_number = (i // batch_size) + 1
#             print(f"📦 Processing batch {current_batch_number}/{total_batches}...")

#             embeddings = embedding_model.embed_documents(batch)

#             vectors = []
#             for j, embedding in enumerate(embeddings):
#                 text = batch[j]
#                 metadata = {
#                     "text": text,
#                     "section": "unknown",
#                     "page": -1,
#                     "source": "",
#                     "type": "paragraph",
#                 }

#                 vectors.append({
#                     "id": f"{namespace}_{i + j}",
#                     "values": embedding,
#                     "metadata": metadata
#                 })

#             print(f"⬆️ Upserting {len(vectors)} vectors from batch {current_batch_number}...")
#             response = index.upsert(vectors=vectors, namespace=namespace)
#             print(f"✅ Upsert for batch {current_batch_number} completed. Response: {response}")
#             total_inserted += len(vectors)

#         return {"status": "success", "inserted": total_inserted}

#     except Exception as e:
#         print(f"❌ Error in embed_and_upsert: {e}")
#         return {"status": "error", "error": str(e)}

# async def retrieve_from_kb(input_params):
#     try:
#         query = input_params.get("query", "")
#         agent_id = input_params.get("agent_id", "")
#         top_k = input_params.get("top_k", 5)

#         if not query:
#             return {"chunks": [], "status": "error", "message": "Query is required"}
#         if not agent_id:
#             return {"chunks": [], "status": "error", "message": "Agent ID is required"}

#         # Get embedding for query
#         query_vector = embedding_model.embed_query(query)

#         # Search in Pinecone using the vector
#         results = index.query(
#             vector=query_vector,
#             namespace=agent_id,
#             top_k=top_k,
#             include_metadata=True
#         )

#         content_blocks = []
#         for match in results.matches:
#             score = match.score
#             if score > 0.0:
#                 metadata = match.metadata or {}
#                 text = metadata.get("text", "")
#                 if text:
#                     content_blocks.append(text)

#         return {"chunks": content_blocks}

#     except Exception as e:
#         print(f"Error in retrieve_from_kb: {e}")
#         return {"chunks": [], "status": "error", "error": str(e)}

# # Function routing
# FUNCTION_HANDLERS = {
#     "retrieve_from_kb": retrieve_from_kb
# }

# FUNCTION_DEFINITIONS = [
#     {
#         "name": "retrieve_from_kb",
#         "description": "Retrieves top-k chunks from the knowledge base using a query and agent_id (namespace).",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The user's search query."
#                 },
#                 "agent_id": {
#                     "type": "string",
#                     "description": "The namespace or agent ID to search in."
#                 },
#                 "top_k": {
#                     "type": "integer",
#                     "description": "Number of top results to return.",
#                     "default": 3
#                 }
#             },
#             "required": ["query", "agent_id"]
#         }
#     }
# ]


import os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from config.settings import settings
import asyncio
from typing import List, Dict, Any

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)

# Initialize HuggingFace embedding model - using API endpoint instead of local
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=settings.HUGGINGFACEHUB_ACCESS_TOKEN,
    task="feature-extraction"
)

# Initialize LLM
def get_llm():
    """Initialize LLM with error handling"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_ACCESS_TOKEN,
            temperature=0.0,
            max_new_tokens=512,
            top_k=1,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise

# Global model instance
_model = None

def get_model():
    global _model
    if _model is None:
        _model = get_llm()
    return _model

def split_text(text: str, chunk_size=500, chunk_overlap=100) -> List[str]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

async def embed_and_upsert(chunks: List[str], namespace: str) -> Dict[str, Any]:
    """Embed chunks and upsert to Pinecone with better error handling"""
    print(f"Embedding and upserting {len(chunks)} chunks into namespace: {namespace}")
    
    if not chunks:
        return {"status": "error", "error": "No chunks provided"}
    
    try:
        batch_size = 50  # Reduced batch size for stability
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        total_inserted = 0

        print(f"🧮 Total batches to process: {total_batches} (batch size = {batch_size})")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            current_batch_number = (i // batch_size) + 1
            print(f"📦 Processing batch {current_batch_number}/{total_batches}...")

            # Filter out empty chunks
            non_empty_batch = [chunk for chunk in batch if chunk.strip()]
            if not non_empty_batch:
                print(f"⚠️ Skipping empty batch {current_batch_number}")
                continue

            try:
                # Use asyncio to handle the embedding with timeout
                embeddings = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: embedding_model.embed_documents(non_empty_batch)
                    ),
                    timeout=60.0  # 60 second timeout
                )
            except asyncio.TimeoutError:
                print(f"⚠️ Embedding timeout for batch {current_batch_number}, retrying...")
                await asyncio.sleep(2)
                try:
                    embeddings = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: embedding_model.embed_documents(non_empty_batch)
                        ),
                        timeout=120.0  # Extended timeout on retry
                    )
                except Exception as e:
                    print(f"❌ Failed to embed batch {current_batch_number}: {e}")
                    continue

            if not embeddings or len(embeddings) != len(non_empty_batch):
                print(f"⚠️ Embedding mismatch for batch {current_batch_number}")
                continue

            vectors = []
            for j, (text, embedding) in enumerate(zip(non_empty_batch, embeddings)):
                if not embedding or len(embedding) == 0:
                    print(f"⚠️ Empty embedding for chunk {j} in batch {current_batch_number}")
                    continue
                    
                metadata = {
                    "text": text[:1000],  # Limit metadata text size
                    "section": "unknown",
                    "page": -1,
                    "source": namespace,
                    "type": "paragraph",
                    "chunk_index": i + j
                }

                vectors.append({
                    "id": f"{namespace}_{i + j}_{current_batch_number}",
                    "values": embedding,
                    "metadata": metadata
                })

            if not vectors:
                print(f"⚠️ No valid vectors in batch {current_batch_number}")
                continue

            print(f"⬆️ Upserting {len(vectors)} vectors from batch {current_batch_number}...")
            
            try:
                response = index.upsert(vectors=vectors, namespace=namespace)
                print(f"✅ Upsert for batch {current_batch_number} completed. Upserted: {response.get('upserted_count', len(vectors))}")
                total_inserted += len(vectors)
                
                # Small delay between batches to avoid rate limits
                if current_batch_number < total_batches:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Upsert failed for batch {current_batch_number}: {e}")
                continue

        print(f"🎉 Completed upserting. Total inserted: {total_inserted}")
        return {"status": "success", "inserted": total_inserted}

    except Exception as e:
        print(f"❌ Error in embed_and_upsert: {e}")
        return {"status": "error", "error": str(e)}

async def retrieve_from_kb(input_params: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve chunks from knowledge base with better error handling"""
    try:
        query = input_params.get("query", "").strip()
        agent_id = input_params.get("agent_id", "").strip()
        top_k = min(input_params.get("top_k", 5), 10)  # Limit top_k

        if not query:
            return {"chunks": [], "status": "error", "message": "Query is required"}
        if not agent_id:
            return {"chunks": [], "status": "error", "message": "Agent ID is required"}

        print(f"🔍 Retrieving for query: '{query[:50]}...' in namespace: '{agent_id}'")

        # Get embedding for query with timeout
        try:
            query_vector = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: embedding_model.embed_query(query)
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print(f"⚠️ Query embedding timeout")
            return {"chunks": [], "status": "error", "message": "Query embedding timeout"}
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            return {"chunks": [], "status": "error", "message": f"Query embedding failed: {str(e)}"}

        if not query_vector or len(query_vector) == 0:
            return {"chunks": [], "status": "error", "message": "Failed to generate query embedding"}

        # Search in Pinecone
        try:
            results = index.query(
                vector=query_vector,
                namespace=agent_id,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
        except Exception as e:
            print(f"❌ Pinecone query failed: {e}")
            return {"chunks": [], "status": "error", "message": f"Vector search failed: {str(e)}"}

        if not results or not hasattr(results, 'matches'):
            return {"chunks": [], "status": "error", "message": "No results from vector search"}

        content_blocks = []
        for match in results.matches:
            try:
                score = getattr(match, 'score', 0.0)
                if score > 0.1:  # Minimum relevance threshold
                    metadata = getattr(match, 'metadata', {}) or {}
                    text = metadata.get("text", "").strip()
                    if text and len(text) > 10:  # Minimum text length
                        content_blocks.append(text)
            except Exception as e:
                print(f"⚠️ Error processing match: {e}")
                continue

        print(f"📄 Retrieved {len(content_blocks)} relevant chunks")
        return {"chunks": content_blocks, "status": "success"}

    except Exception as e:
        print(f"❌ Error in retrieve_from_kb: {e}")
        return {"chunks": [], "status": "error", "error": str(e)}

# Function routing
FUNCTION_HANDLERS = {
    "retrieve_from_kb": retrieve_from_kb
}

FUNCTION_DEFINITIONS = [
    {
        "name": "retrieve_from_kb",
        "description": "Retrieves top-k chunks from the knowledge base using a query and agent_id (namespace).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's search query."
                },
                "agent_id": {
                    "type": "string",
                    "description": "The namespace or agent ID to search in."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query", "agent_id"]
        }
    }
]
