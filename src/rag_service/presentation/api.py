import os

from fastapi import FastAPI, UploadFile, File
from rag_service.infrastructure.pdf_parser import PymupdfParser
from rag_service.infrastructure.simple_chunker import SimpleChunker
from rag_service.infrastructure.embedding import SentenceTransformerEmbeddingService
from rag_service.infrastructure.memory_document_repo import MemoryDocumentRepository    
from rag_service.infrastructure.memory_vector_repo import MemoryVectorRepository
from rag_service.infrastructure.llm_service import OllamaLLMService

from rag_service.use_cases.ingest_pdf_use_case import IngestPdfUseCase
from rag_service.use_cases.ask_question_use_case import AskQuestionUseCase

from rag_service.domain.models import Question

app = FastAPI()

pdf_parser = PymupdfParser()
chunker = SimpleChunker()
embedder = SentenceTransformerEmbeddingService(model_name="all-MiniLM-L6-v2")
document_repo = MemoryDocumentRepository()
vector_repo = MemoryVectorRepository()
llm_service = OllamaLLMService(model_name="llama3.1:8b")

ingest_use_case = IngestPdfUseCase(
    pdf_parser, 
    chunker, 
    embedder, 
    document_repo, 
    vector_repo
)

ask_question_use_case = AskQuestionUseCase(
    embedder, 
    vector_repo, 
    llm_service
)

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
    answer = ask_question_use_case.execute(question, top_k=3)
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
