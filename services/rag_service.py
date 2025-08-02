# from pinecone import Pinecone
# from services.vector_store import embed_and_upsert, retrieve_from_kb, split_text
# from services.pdf_parser import extract_text_from_pdf
# from services.hf_model import ask_gpt
# import re
# import tempfile
# import aiohttp
# import asyncio
# import os
# from config.settings import settings

# # Initialize Pinecone with error handling
# try:
#     pc = Pinecone(api_key=settings.PINECONE_API_KEY)
#     index = pc.Index(settings.PINECONE_INDEX_NAME)
# except Exception as e:
#     print(f"Error initializing Pinecone: {e}")
#     raise

# def generate_namespace_from_url(url: str) -> str:
#     return re.sub(r'\W+', '_', url).strip('_').lower()

# async def download_pdf_to_temp_file(pdf_url: str) -> str:
#     timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
#     try:
#         async with aiohttp.ClientSession(timeout=timeout) as session:
#             async with session.get(pdf_url) as response:
#                 if response.status != 200:
#                     raise Exception(f"Failed to download PDF. Status: {response.status}")
#                 content = await response.read()
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                     tmp.write(content)
#                     return tmp.name
#     except asyncio.TimeoutError:
#         raise Exception("PDF download timed out after 30 seconds")
#     except Exception as e:
#         raise Exception(f"Failed to download PDF: {str(e)}")

# async def process_documents_and_questions(pdf_url: str, questions: list[str]) -> dict:
#     print(f"Processing documents from URL: {pdf_url}")
    
#     try:
#         # Step 1: Generate namespace from URL and check existence
#         agent_id = generate_namespace_from_url(pdf_url)
        
#         # Add timeout for Pinecone operations
#         try:
#             stats = index.describe_index_stats()
#             existing_namespaces = stats.namespaces.keys() if stats.namespaces else []
#         except Exception as e:
#             print(f"Error checking existing namespaces: {e}")
#             existing_namespaces = []

#         # Step 2: If namespace does not exist, process and upsert
#         if agent_id not in existing_namespaces:
#             print(f"🆕 Namespace '{agent_id}' not found. Proceeding with PDF download and embedding...")

#             try:
#                 local_pdf_path = await download_pdf_to_temp_file(pdf_url)
#                 raw_text_output = extract_text_from_pdf(local_pdf_path)
                
#                 # Clean up temp file
#                 try:
#                     os.unlink(local_pdf_path)
#                 except:
#                     pass
                
#                 raw_text = "\n".join(raw_text_output) if isinstance(raw_text_output, list) else raw_text_output
#                 print(f"📄 Extracted {len(raw_text)} characters from PDF")

#                 if not raw_text.strip():
#                     raise Exception("No text could be extracted from the PDF")

#                 chunks = split_text(raw_text)
#                 print(f"🧾 Extracted {len(chunks)} chunks from PDF")
#                 usable_chunks = [c for c in chunks if len(c.strip()) > 50]
#                 print(f"✅ Usable chunks (>50 chars): {len(usable_chunks)}")

#                 if not usable_chunks:
#                     raise Exception("No usable text chunks found in PDF")

#                 await embed_and_upsert(usable_chunks, agent_id)
#             except Exception as e:
#                 print(f"Error processing PDF: {e}")
#                 raise Exception(f"Failed to process PDF: {str(e)}")
#         else:
#             print(f"📂 Namespace '{agent_id}' already exists. Skipping download and embedding.")

#         # Step 3: Parallel question processing with reduced concurrency
#         semaphore = asyncio.Semaphore(3)  # Reduced from 10 to 3 to avoid rate limits

#         async def process_question(index: int, question: str) -> tuple[int, str, str]:
#             async with semaphore:
#                 for attempt in range(3):
#                     try:
#                         retrieval_input = {"query": question, "agent_id": agent_id, "top_k": 3}
#                         retrieved = await retrieve_from_kb(retrieval_input)
#                         retrieved_chunks = retrieved.get("chunks", [])
                        
#                         if not retrieved_chunks:
#                             print(f"⚠️ Q{index}: No chunks retrieved for question: {question[:50]}...")
#                             return (index, question, "I couldn't find relevant information to answer this question.")

#                         max_context_chars = 3000
#                         context = "\n".join(retrieved_chunks)[:max_context_chars]

#                         print(f"✏️ Q{index}: Context preview: {context[:100]}...")
#                         answer = await ask_gpt(context, question)
#                         return (index, question, answer)
                        
#                     except Exception as e:
#                         print(f"⚠️ Q{index}: Attempt {attempt + 1} failed with error: {e}")
#                         if attempt < 2:  # Don't sleep on last attempt
#                             await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
#                 return (index, question, "Sorry, I couldn't find relevant information to answer this question.")

#         print(f"🧠 Parallel processing {len(questions)} questions...")
        
#         if not questions:
#             return {}
            
#         # Add timeout for question processing
#         try:
#             tasks = [asyncio.create_task(process_question(i, q)) for i, q in enumerate(questions)]
#             responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)  # 2 minute timeout
#         except asyncio.TimeoutError:
#             print("⚠️ Question processing timed out")
#             raise Exception("Processing timed out. Please try with fewer questions or a smaller document.")

#         # Step 4: Return sorted results
#         results = {q: ans for _, q, ans in sorted(responses)}
#         return results
        
#     except Exception as e:
#         print(f"❌ Error in process_documents_and_questions: {e}")
#         raise

from pinecone import Pinecone
from services.vector_store import embed_and_upsert, retrieve_from_kb, split_text
from services.pdf_parser import extract_text_from_pdf
from services.hf_model import ask_gpt
import re
import tempfile
import aiohttp
import asyncio
import os
import logging
from typing import List, Dict, Any, Tuple
from config.settings import settings

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    logger.info("✅ Pinecone initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing Pinecone: {e}")
    raise

def generate_namespace_from_url(url: str) -> str:
    """Generate a clean namespace from URL"""
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Remove protocol and clean up the URL
    clean_url = re.sub(r'^https?://', '', url.lower())
    # Replace non-alphanumeric characters with underscores
    namespace = re.sub(r'[^\w]+', '_', clean_url).strip('_')
    # Limit length to avoid issues
    if len(namespace) > 50:
        namespace = namespace[:50]
    
    logger.info(f"Generated namespace: {namespace} from URL: {url}")
    return namespace

async def download_pdf_to_temp_file(pdf_url: str) -> str:
    """Download PDF from URL to temporary file with enhanced error handling"""
    if not pdf_url or not pdf_url.strip():
        raise ValueError("PDF URL cannot be empty")
    
    # Validate URL format
    if not pdf_url.startswith(('http://', 'https://')):
        raise ValueError("PDF URL must start with http:// or https://")
    
    timeout = aiohttp.ClientTimeout(total=60, connect=30)  # Increased timeout
    
    try:
        logger.info(f"📥 Downloading PDF from: {pdf_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(pdf_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF. Status: {response.status}, Reason: {response.reason}")
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
                    logger.warning(f"⚠️ Content type is {content_type}, but proceeding anyway")
                
                content = await response.read()
                
                if len(content) == 0:
                    raise Exception("Downloaded PDF file is empty")
                
                # Validate PDF signature
                if not content.startswith(b'%PDF'):
                    logger.warning("⚠️ File doesn't have PDF signature, but proceeding anyway")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    temp_path = tmp.name
                
                logger.info(f"✅ PDF downloaded successfully to: {temp_path} ({len(content)} bytes)")
                return temp_path
                
    except asyncio.TimeoutError:
        raise Exception("PDF download timed out after 60 seconds")
    except aiohttp.ClientError as e:
        raise Exception(f"Network error downloading PDF: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

async def process_documents_and_questions(pdf_url: str, questions: List[str]) -> Dict[str, str]:
    """
    Main function to process documents and answer questions
    """
    if not pdf_url or not pdf_url.strip():
        raise ValueError("PDF URL is required")
    
    if not questions or len(questions) == 0:
        raise ValueError("At least one question is required")
    
    # Limit number of questions to prevent abuse
    if len(questions) > 20:
        raise ValueError("Maximum 20 questions allowed per request")
    
    logger.info(f"🚀 Processing documents from URL: {pdf_url}")
    logger.info(f"📋 Number of questions: {len(questions)}")
    
    try:
        # Step 1: Generate namespace from URL and check existence
        agent_id = generate_namespace_from_url(pdf_url)
        
        # Check existing namespaces with timeout
        existing_namespaces = []
        try:
            stats = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: index.describe_index_stats()
                ),
                timeout=15.0
            )
            existing_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            logger.info(f"📊 Found {len(existing_namespaces)} existing namespaces")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout checking existing namespaces, proceeding with processing")
        except Exception as e:
            logger.warning(f"⚠️ Error checking existing namespaces: {e}, proceeding with processing")

        # Step 2: Process PDF if namespace doesn't exist
        if agent_id not in existing_namespaces:
            logger.info(f"🆕 Namespace '{agent_id}' not found. Processing PDF...")
            
            local_pdf_path = None
            try:
                # Download PDF
                local_pdf_path = await download_pdf_to_temp_file(pdf_url)
                
                # Extract text from PDF
                raw_text_output = extract_text_from_pdf(local_pdf_path)
                
                # Handle different return types from extract_text_from_pdf
                if isinstance(raw_text_output, list):
                    if raw_text_output and hasattr(raw_text_output[0], 'page_content'):
                        # List of Document objects
                        raw_text = "\n\n".join([doc.page_content for doc in raw_text_output])
                    else:
                        # List of strings
                        raw_text = "\n\n".join(raw_text_output)
                else:
                    # Single string
                    raw_text = str(raw_text_output)
                
                logger.info(f"📄 Extracted {len(raw_text)} characters from PDF")

                if not raw_text or not raw_text.strip():
                    raise Exception("No text could be extracted from the PDF")

                # Split text into chunks
                chunks = split_text(raw_text)
                logger.info(f"🧾 Split into {len(chunks)} initial chunks")
                
                # Filter usable chunks
                usable_chunks = [c.strip() for c in chunks if c and len(c.strip()) > 50]
                logger.info(f"✅ Filtered to {len(usable_chunks)} usable chunks (>50 chars)")

                if not usable_chunks:
                    raise Exception("No usable text chunks found in PDF after filtering")

                # Embed and upsert chunks
                result = await embed_and_upsert(usable_chunks, agent_id)
                if result.get("status") != "success":
                    raise Exception(f"Failed to embed and upsert chunks: {result.get('error', 'Unknown error')}")
                    
                logger.info(f"✅ Successfully processed and embedded {result.get('inserted', 0)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Error processing PDF: {e}")
                raise Exception(f"Failed to process PDF: {str(e)}")
            finally:
                # Clean up temp file
                if local_pdf_path and os.path.exists(local_pdf_path):
                    try:
                        os.unlink(local_pdf_path)
                        logger.info("🗑️ Cleaned up temporary PDF file")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")
        else:
            logger.info(f"📂 Namespace '{agent_id}' already exists. Skipping PDF processing.")

        # Step 3: Process questions with controlled concurrency
        semaphore = asyncio.Semaphore(2)  # Reduced to 2 for stability
        
        async def process_question(q_index: int, question: str) -> Tuple[int, str, str]:
            """Process a single question with retry logic"""
            if not question or not question.strip():
                return (q_index, question, "Question cannot be empty.")
            
            question = question.strip()
            if len(question) > 500:
                question = question[:500] + "..."
            
            async with semaphore:
                for attempt in range(3):
                    try:
                        logger.info(f"🔍 Q{q_index+1}: Processing question (attempt {attempt+1})")
                        
                        # Retrieve relevant chunks
                        retrieval_input = {
                            "query": question, 
                            "agent_id": agent_id, 
                            "top_k": 5
                        }
                        
                        retrieved = await retrieve_from_kb(retrieval_input)
                        
                        if retrieved.get("status") == "error":
                            raise Exception(f"Retrieval failed: {retrieved.get('message', 'Unknown error')}")
                        
                        retrieved_chunks = retrieved.get("chunks", [])
                        
                        if not retrieved_chunks:
                            logger.warning(f"⚠️ Q{q_index+1}: No chunks retrieved for: {question[:50]}...")
                            return (q_index, question, "I couldn't find relevant information in the document to answer this question.")

                        # Prepare context with size limit
                        max_context_chars = 2500  # Reduced for better performance
                        context = "\n\n".join(retrieved_chunks)[:max_context_chars]
                        
                        if len(context) >= max_context_chars:
                            context += "\n\n[Content truncated...]"

                        logger.info(f"📝 Q{q_index+1}: Retrieved {len(retrieved_chunks)} chunks, context length: {len(context)}")
                        
                        # Get answer from language model
                        answer = await ask_gpt(context, question)
                        
                        if not answer or answer.strip() == "":
                            answer = "I couldn't generate a proper answer to this question."
                        
                        logger.info(f"✅ Q{q_index+1}: Answer generated successfully")
                        return (q_index, question, answer.strip())
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Q{q_index+1}: Attempt {attempt + 1} failed: {e}")
                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(min(2 ** attempt, 5))  # Cap backoff at 5 seconds
                
                # All attempts failed
                logger.error(f"❌ Q{q_index+1}: All attempts failed")
                return (q_index, question, "Sorry, I encountered an error while processing this question. Please try again.")

        # Process all questions
        logger.info(f"🧠 Processing {len(questions)} questions in parallel...")
        
        try:
            tasks = [
                asyncio.create_task(process_question(i, q)) 
                for i, q in enumerate(questions)
            ]
            
            # Set timeout based on number of questions
            timeout_seconds = min(180, 30 + len(questions) * 15)  # 30s base + 15s per question, max 3 minutes
            
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=timeout_seconds
            )
            
            # Handle any exceptions in responses
            final_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"❌ Q{i+1}: Exception occurred: {response}")
                    final_responses.append((i, questions[i], f"Error processing question: {str(response)}"))
                else:
                    final_responses.append(response)
            
        except asyncio.TimeoutError:
            logger.error(f"⚠️ Question processing timed out after {timeout_seconds}s")
            raise Exception(f"Processing timed out after {timeout_seconds} seconds. Please try with fewer questions or simpler queries.")

        # Step 4: Return sorted results
        results = {q: ans for _, q, ans in sorted(final_responses)}
        
        logger.info(f"🎉 Successfully processed all {len(results)} questions")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in process_documents_and_questions: {e}")
        raise
