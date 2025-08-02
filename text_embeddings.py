#!/usr/bin/env python3
"""
Script to test the embedded data in Pinecone
Usage: python test_embeddings.py
"""

import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load environment variables
load_dotenv()

class EmbeddingTester:
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

    def check_index_stats(self):
        """Check overall index statistics"""
        try:
            stats = self.index.describe_index_stats()
            print("\n📊 Index Statistics:")
            print(f"  Total vectors: {stats.total_vector_count}")
            print(f"  Index fullness: {stats.index_fullness}")
            print(f"  Dimension: {stats.dimension}")
            
            if stats.namespaces:
                print("  Namespaces:")
                for namespace, ns_stats in stats.namespaces.items():
                    print(f"    - {namespace}: {ns_stats.vector_count} vectors")
            else:
                print("  No namespaces found")
                
            return stats
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return None

    async def test_search(self, namespace: str, query: str, top_k: int = 5):
        """Test search functionality"""
        try:
            print(f"\n🔍 Testing search in namespace '{namespace}'")
            print(f"Query: '{query}'")
            
            # Get embedding for query
            query_vector = self.embedding_model.embed_query(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_vector,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )
            
            print(f"📋 Found {len(results.matches)} results:")
            
            for i, match in enumerate(results.matches, 1):
                score = match.score
                metadata = match.metadata or {}
                text = metadata.get("text", "No text available")
                
                print(f"\n  Result {i}:")
                print(f"    Score: {score:.4f}")
                print(f"    ID: {match.id}")
                print(f"    Text (first 200 chars): {text[:200]}...")
                if len(text) > 200:
                    print(f"    [Text truncated, full length: {len(text)} chars]")
                    
            return results
            
        except Exception as e:
            print(f"❌ Error testing search: {e}")
            return None

    async def interactive_test(self, namespace: str = "extracted_text_embedding"):
        """Interactive testing session"""
        print(f"\n🎯 Interactive testing for namespace: {namespace}")
        print("Enter search queries (type 'quit' to exit):")
        
        while True:
            try:
                query = input("\n🔍 Enter query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not query:
                    print("Please enter a valid query")
                    continue
                    
                await self.test_search(namespace, query, top_k=3)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Main function"""
    print("🚀 Starting embedding test...")
    
    # Check for required environment variables
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "HUGGINGFACEHUB_ACCESS_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return

    # Initialize tester
    try:
        tester = EmbeddingTester()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return

    # Check index stats
    stats = tester.check_index_stats()
    if not stats or stats.total_vector_count == 0:
        print("\n⚠️ No vectors found in index. Please run embed_text_file.py first.")
        return

    # Test with some sample queries
    namespace = "extracted_text_embedding"
    sample_queries = [
        "What is the waiting period for cataract surgery?"
    ]
    
    print("\n🧪 Running sample queries...")
    for query in sample_queries:
        await tester.test_search(namespace, query, top_k=2)
        await asyncio.sleep(1)  # Small delay between queries
    
    # Interactive testing
    try:
        await tester.interactive_test(namespace)
    except KeyboardInterrupt:
        print("\n👋 Test session ended")

if __name__ == "__main__":
    asyncio.run(main())