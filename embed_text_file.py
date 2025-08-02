#!/usr/bin/env python3
"""
Independent script to embed extracted_text.txt into Pinecone
Usage: python embed_text_file.py
"""

import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load environment variables
load_dotenv()

class TextEmbedder:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        
        # Initialize HuggingFace embedding model
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        )
        
        print("✅ Initialized Pinecone and HuggingFace embedding model")

    def load_text_file(self, file_path: str) -> str:
        """Load text from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"📄 Loaded {len(text)} characters from {file_path}")
            return text
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            raise

    def split_text(self, text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
        """Split text into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        print(f"🧾 Split text into {len(chunks)} chunks")
        return chunks

    async def embed_and_upsert(self, chunks: list[str], namespace: str):
        """Embed chunks and upsert to Pinecone"""
        print(f"🚀 Embedding and upserting {len(chunks)} chunks into namespace: {namespace}")
        
        try:
            batch_size = 50  # Reduced batch size for stability
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            total_inserted = 0

            print(f"🧮 Total batches to process: {total_batches} (batch size = {batch_size})")

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                current_batch_number = (i // batch_size) + 1
                print(f"📦 Processing batch {current_batch_number}/{total_batches}...")

                # Generate embeddings for the batch
                print(f"🧠 Generating embeddings for batch {current_batch_number}...")
                embeddings = self.embedding_model.embed_documents(batch)
                print(f"✅ Generated {len(embeddings)} embeddings")

                # Prepare vectors for upsert
                vectors = []
                for j, embedding in enumerate(embeddings):
                    text = batch[j]
                    metadata = {
                        "text": text,
                        "section": "extracted_text",
                        "chunk_id": i + j,
                        "source": "extracted_text.txt",
                        "type": "paragraph",
                        "length": len(text)
                    }

                    vectors.append({
                        "id": f"{namespace}_{i + j}",
                        "values": embedding,
                        "metadata": metadata
                    })

                # Upsert to Pinecone
                print(f"⬆️ Upserting {len(vectors)} vectors from batch {current_batch_number}...")
                response = self.index.upsert(vectors=vectors, namespace=namespace)
                print(f"✅ Upsert for batch {current_batch_number} completed. Response: {response}")
                total_inserted += len(vectors)

                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)

            print(f"🎉 Successfully inserted {total_inserted} vectors into namespace '{namespace}'")
            return {"status": "success", "inserted": total_inserted}

        except Exception as e:
            print(f"❌ Error in embed_and_upsert: {e}")
            return {"status": "error", "error": str(e)}

    def check_existing_vectors(self, namespace: str):
        """Check if vectors already exist in the namespace"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.namespaces or {}
            
            if namespace in namespaces:
                vector_count = namespaces[namespace].vector_count
                print(f"📊 Namespace '{namespace}' already has {vector_count} vectors")
                return vector_count
            else:
                print(f"📊 Namespace '{namespace}' does not exist yet")
                return 0
        except Exception as e:
            print(f"⚠️ Error checking existing vectors: {e}")
            return 0

    async def process_file(self, file_path: str, namespace: str = None, force_reprocess: bool = False):
        """Main processing function"""
        if namespace is None:
            namespace = "extracted_text_embedding"
        
        print(f"🎯 Processing file: {file_path}")
        print(f"🎯 Target namespace: {namespace}")
        
        # Check if vectors already exist
        if not force_reprocess:
            existing_count = self.check_existing_vectors(namespace)
            if existing_count > 0:
                response = input(f"Namespace '{namespace}' already has {existing_count} vectors. "
                               f"Do you want to proceed anyway? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Aborted by user")
                    return

        # Load and process text
        text = self.load_text_file(file_path)
        
        if not text.strip():
            print("❌ File is empty or contains no readable text")
            return

        # Split into chunks
        chunks = self.split_text(text)
        
        if not chunks:
            print("❌ No chunks generated from text")
            return

        # Filter out very short chunks
        usable_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        print(f"✅ Usable chunks (>50 chars): {len(usable_chunks)}")
        
        if not usable_chunks:
            print("❌ No usable chunks found")
            return

        # Embed and upsert
        result = await self.embed_and_upsert(usable_chunks, namespace)
        
        if result["status"] == "success":
            print(f"🎉 Successfully processed {file_path}")
            print(f"📊 Total vectors inserted: {result['inserted']}")
            print(f"🏷️ Namespace: {namespace}")
        else:
            print(f"❌ Failed to process file: {result.get('error', 'Unknown error')}")

async def main():
    """Main function"""
    print("🚀 Starting text file embedding process...")
    
    # Check for required environment variables
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "HUGGINGFACEHUB_ACCESS_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please make sure your .env file contains these variables")
        return

    # Initialize embedder
    try:
        embedder = TextEmbedder()
    except Exception as e:
        print(f"❌ Failed to initialize embedder: {e}")
        return

    # Check if file exists
    file_path = "extracted_text.txt"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please make sure extracted_text.txt is in the current directory")
        return

    # Process the file
    try:
        await embedder.process_file(
            file_path=file_path,
            namespace="extracted_text_embedding",
            force_reprocess=False
        )
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())