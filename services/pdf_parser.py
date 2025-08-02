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
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

def inject_markdown_headers(text: str) -> str:
    """
    Tag waiting period and medical clauses using markdown headers.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if re.search(r"(one|two|three|\d+)[-\s]?(year)?s?\s+waiting period", line, re.IGNORECASE):
            result.append(f"# {line}")
        elif re.match(r"^[a-zA-Z]\.\s+.+", line):
            result.append(f"### {line}")
        else:
            result.append(line)
    return "\n".join(result)

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF from disk, parse, tag, and split it into semantically meaningful chunks.
    Returns a list of LangChain Document objects.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_name = os.path.basename(pdf_path)
    
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file_name
            if "page_number" not in doc.metadata:
                doc.metadata["page_number"] = None
    except Exception as e:
        print(f"❌ Failed to load PDF: {e}")
        return []

    combined_text = "\n\n".join([doc.page_content for doc in docs])
    markdown_text = inject_markdown_headers(combined_text)

    headers_to_split_on = [("#", "section"), ("###", "clause")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    try:
        md_chunks = md_splitter.split_text(markdown_text)
        if md_chunks and len(md_chunks) >= 5:
            final_chunks = [
                Document(
                    page_content=chunk.page_content.strip(),
                    metadata={
                        "source_file": file_name,
                        "page_number": None
                    }
                )
                for chunk in md_chunks
            ]
            print(f"📄 Extracted {len(final_chunks)} markdown chunks from PDF")
            print(f"🔍 First chunk (100 chars): {repr(final_chunks[0].page_content[:100])}")
            return final_chunks
        else:
            print("⚠️ Too few markdown chunks. Falling back to char-based splitting.")
    except Exception as e:
        print(f"⚠️ Markdown splitting failed: {e}")

    # === Fallback to recursive char splitter ===
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fallback_chunks = char_splitter.split_documents(docs)
    print(f"📄 Extracted {len(fallback_chunks)} fallback character-based chunks from PDF")
    print(f"🔍 First chunk (100 chars): {repr(fallback_chunks[0].page_content[:100])}")
    return fallback_chunks