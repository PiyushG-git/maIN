# import pdfplumber
# import httpx
# import tempfile

# async def download_pdf(url: str) -> str:
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         response.raise_for_status()

#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file.write(response.content)
#             print(f"✅ Downloaded PDF to: {temp_file.name}")
#             return temp_file.name

# def extract_text_from_pdf(pdf_path: str) -> list[str]:
#     full_text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             if text:
#                 full_text += text + "\n"
#             else:
#                 print(f"⚠️ No text found on page {i+1}")

#     chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
    
#     print(f"📄 Extracted {len(chunks)} chunks from PDF using pdfplumber")
#     print(f"🔍 First chunk (100 chars): {repr(chunks[0][:100]) if chunks else 'No chunks'}")
    
#     return chunks

import os
import re
import tempfile
from typing import List, Union
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

# Set up logging
logger = logging.getLogger(__name__)

def inject_markdown_headers(text: str) -> str:
    """
    Tag waiting period and medical clauses using markdown headers.
    """
    if not text or not text.strip():
        return text
        
    try:
        lines = text.split("\n")
        result = []
        
        for line in lines:
            line = line.strip()
            if not line:
                result.append(line)
                continue
                
            # Look for waiting period patterns
            if re.search(r"(one|two|three|\d+)[-\s]?(year)?s?\s+waiting period", line, re.IGNORECASE):
                result.append(f"# {line}")
            # Look for clause patterns (letter followed by period and text)
            elif re.match(r"^[a-zA-Z]\.\s+.+", line):
                result.append(f"### {line}")
            else:
                result.append(line)
                
        return "\n".join(result)
    except Exception as e:
        logger.warning(f"Error injecting markdown headers: {e}")
        return text

def extract_text_from_pdf(pdf_path: str) -> List[Union[Document, str]]:
    """
    Load a PDF from disk, parse, tag, and split it into semantically meaningful chunks.
    Returns a list of LangChain Document objects or strings as fallback.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_name = os.path.basename(pdf_path)
    logger.info(f"Processing PDF: {file_name}")
    
    try:
        # Load PDF using PyMuPDF
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content could be extracted from PDF")

        # Add metadata to documents
        for i, doc in enumerate(docs):
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source_file"] = file_name
            if "page_number" not in doc.metadata:
                doc.metadata["page_number"] = i + 1
                
        logger.info(f"Loaded {len(docs)} pages from PDF")

    except Exception as e:
        logger.error(f"Failed to load PDF with PyMuPDF: {e}")
        
        # Fallback to pdfplumber if available
        try:
            import pdfplumber
            
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content.append(page_text.strip())
                        else:
                            logger.warning(f"No text found on page {i+1}")
                    except Exception as page_error:
                        logger.warning(f"Error extracting text from page {i+1}: {page_error}")
                        continue
            
            if not text_content:
                raise ValueError("No text could be extracted from PDF using fallback method")
            
            # Convert to Document objects
            docs = []
            for i, text in enumerate(text_content):
                doc = Document(
                    page_content=text,
                    metadata={
                        "source_file": file_name,
                        "page_number": i + 1
                    }
                )
                docs.append(doc)
                
            logger.info(f"Extracted text from {len(docs)} pages using pdfplumber fallback")
            
        except ImportError:
            logger.error("pdfplumber not available for fallback")
            raise Exception("Failed to extract text from PDF and no fallback available")
        except Exception as fallback_error:
            logger.error(f"Fallback PDF extraction failed: {fallback_error}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    # Combine all text content
    try:
        combined_text = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
        
        if not combined_text.strip():
            raise ValueError("No text content found in PDF")
            
        logger.info(f"Combined text length: {len(combined_text)} characters")
        
        # Apply markdown header injection
        markdown_text = inject_markdown_headers(combined_text)
        
        # Try markdown-based splitting first
        headers_to_split_on = [("#", "section"), ("###", "clause")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

        try:
            md_chunks = md_splitter.split_text(markdown_text)
            
            if md_chunks and len(md_chunks) >= 3:
                final_chunks = []
                for chunk in md_chunks:
                    if hasattr(chunk, 'page_content'):
                        content = chunk.page_content.strip()
                    else:
                        content = str(chunk).strip()
                        
                    if content and len(content) > 50:  # Minimum chunk size
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source_file": file_name,
                                "page_number": None,
                                "chunk_type": "markdown"
                            }
                        )
                        final_chunks.append(doc)
                
                if final_chunks:
                    logger.info(f"Successfully created {len(final_chunks)} markdown chunks")
                    logger.info(f"First chunk preview: {final_chunks[0].page_content[:100]}...")
                    return final_chunks
                else:
                    logger.warning("No valid markdown chunks created, falling back to character splitting")
            else:
                logger.warning("Too few markdown chunks, falling back to character splitting")
                
        except Exception as md_error:
            logger.warning(f"Markdown splitting failed: {md_error}, falling back to character splitting")

    except Exception as text_error:
        logger.error(f"Error processing combined text: {text_error}")
        # Return original docs if text processing fails
        return docs

    # Fallback to recursive character splitter
    try:
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        fallback_chunks = char_splitter.split_documents(docs)
        
        # Filter out very small chunks
        filtered_chunks = [
            chunk for chunk in fallback_chunks 
            if chunk.page_content and len(chunk.page_content.strip()) > 50
        ]
        
        if not filtered_chunks:
            # Last resort: return simple text chunks
            simple_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            text_chunks = text_splitter.split_text(combined_text)
            
            for i, text_chunk in enumerate(text_chunks):
                if text_chunk.strip() and len(text_chunk.strip()) > 50:
                    simple_chunks.append(text_chunk.strip())
            
            logger.info(f"Created {len(simple_chunks)} simple text chunks as final fallback")
            return simple_chunks
        
        # Add chunk type to metadata
        for chunk in filtered_chunks:
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["chunk_type"] = "character"
            
        logger.info(f"Successfully created {len(filtered_chunks)} character-based chunks")
        logger.info(f"First chunk preview: {filtered_chunks[0].page_content[:100]}...")
        return filtered_chunks
        
    except Exception as char_error:
        logger.error(f"Character splitting also failed: {char_error}")
        raise Exception(f"All text splitting methods failed. PDF may be corrupted or unreadable.")
