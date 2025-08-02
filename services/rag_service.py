from pinecone import Pinecone
from services.vector_store import embed_and_upsert, retrieve_from_kb, split_text
from services.pdf_parser import extract_text_from_pdf
from services.hf_model import ask_gpt
import re
import tempfile
import aiohttp
import asyncio
import os
from config.settings import settings

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

def generate_namespace_from_url(url: str) -> str:
    return re.sub(r'\W+', '_', url).strip('_').lower()

async def download_pdf_to_temp_file(pdf_url: str) -> str:
    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF. Status: {response.status}")
                content = await response.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    return tmp.name
    except asyncio.TimeoutError:
        raise Exception("PDF download timed out after 30 seconds")
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

async def process_documents_and_questions(pdf_url: str, questions: list[str]) -> dict:
    print(f"Processing documents from URL: {pdf_url}")
    
    try:
        # Step 1: Generate namespace from URL and check existence
        agent_id = generate_namespace_from_url(pdf_url)
        
        # Add timeout for Pinecone operations
        try:
            stats = index.describe_index_stats()
            existing_namespaces = stats.namespaces.keys() if stats.namespaces else []
        except Exception as e:
            print(f"Error checking existing namespaces: {e}")
            existing_namespaces = []

        # Step 2: If namespace does not exist, process and upsert
        if agent_id not in existing_namespaces:
            print(f"🆕 Namespace '{agent_id}' not found. Proceeding with PDF download and embedding...")

            try:
                local_pdf_path = await download_pdf_to_temp_file(pdf_url)
                raw_text_output = extract_text_from_pdf(local_pdf_path)
                
                # Clean up temp file
                try:
                    os.unlink(local_pdf_path)
                except:
                    pass
                
                raw_text = "\n".join(raw_text_output) if isinstance(raw_text_output, list) else raw_text_output
                print(f"📄 Extracted {len(raw_text)} characters from PDF")

                if not raw_text.strip():
                    raise Exception("No text could be extracted from the PDF")

                chunks = split_text(raw_text)
                print(f"🧾 Extracted {len(chunks)} chunks from PDF")
                usable_chunks = [c for c in chunks if len(c.strip()) > 50]
                print(f"✅ Usable chunks (>50 chars): {len(usable_chunks)}")

                if not usable_chunks:
                    raise Exception("No usable text chunks found in PDF")

                await embed_and_upsert(usable_chunks, agent_id)
            except Exception as e:
                print(f"Error processing PDF: {e}")
                raise Exception(f"Failed to process PDF: {str(e)}")
        else:
            print(f"📂 Namespace '{agent_id}' already exists. Skipping download and embedding.")

        # Step 3: Parallel question processing with reduced concurrency
        semaphore = asyncio.Semaphore(3)  # Reduced from 10 to 3 to avoid rate limits

        async def process_question(index: int, question: str) -> tuple[int, str, str]:
            async with semaphore:
                for attempt in range(3):
                    try:
                        retrieval_input = {"query": question, "agent_id": agent_id, "top_k": 3}
                        retrieved = await retrieve_from_kb(retrieval_input)
                        retrieved_chunks = retrieved.get("chunks", [])
                        
                        if not retrieved_chunks:
                            print(f"⚠️ Q{index}: No chunks retrieved for question: {question[:50]}...")
                            return (index, question, "I couldn't find relevant information to answer this question.")

                        max_context_chars = 3000
                        context = "\n".join(retrieved_chunks)[:max_context_chars]

                        print(f"✏️ Q{index}: Context preview: {context[:100]}...")
                        answer = await ask_gpt(context, question)
                        return (index, question, answer)
                        
                    except Exception as e:
                        print(f"⚠️ Q{index}: Attempt {attempt + 1} failed with error: {e}")
                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                return (index, question, "Sorry, I couldn't find relevant information to answer this question.")

        print(f"🧠 Parallel processing {len(questions)} questions...")
        
        if not questions:
            return {}
            
        # Add timeout for question processing
        try:
            tasks = [asyncio.create_task(process_question(i, q)) for i, q in enumerate(questions)]
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)  # 2 minute timeout
        except asyncio.TimeoutError:
            print("⚠️ Question processing timed out")
            raise Exception("Processing timed out. Please try with fewer questions or a smaller document.")

        # Step 4: Return sorted results
        results = {q: ans for _, q, ans in sorted(responses)}
        return results
        
    except Exception as e:
        print(f"❌ Error in process_documents_and_questions: {e}")
        raise