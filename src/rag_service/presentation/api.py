import os

from fastapi import FastAPI, UploadFile, File
from src.rag_service.infrastructure.pdf_parser import PymupdfParser
from src.rag_service.infrastructure.simple_chunker import SimpleChunker
from src.rag_service.infrastructure.embedding import SentenceTransformerEmbeddingService
from src.rag_service.infrastructure.memory_document_repo import MemoryDocumentRepository    
from src.rag_service.infrastructure.memory_vector_repo import MemoryVectorRepository
from src.rag_service.infrastructure.llm_service import OllamaLLMService

from src.rag_service.use_cases.ingest_pdf_use_case import IngestPdfUseCase
from src.rag_service.use_cases.ask_question_use_case import AskQuestionUseCase

from src.rag_service.domain.models import Question

from src.dependency_injector import DependencyInjector

app = FastAPI()

dependency_injector = DependencyInjector()

ingest_use_case = dependency_injector.create_ingest_pdf_use_case()
ask_question_use_case = dependency_injector.create_ask_question_use_case()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # copy into
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    document_id = ingest_use_case.execute(file_path)
    return {
        "document_id": document_id,
        "filename": file.filename
    }

@app.post("/ask")
async def ask_question(question: Question):
    answer = ask_question_use_case.execute(question)
    return {
        "answer": answer.text,
        "used_chunks": [
            {
                "page_number": chunk.page_number,
                "text": chunk.text[:100]
            }
            for chunk in answer.used_chunks
        ]
    }
