import os

from fastapi import FastAPI, File, UploadFile

from src.rag_service.domain.models import Question
from src.rag_service.use_cases.ask_question_use_case import AskQuestionUseCase
from src.rag_service.use_cases.ingest_pdf_use_case import IngestPdfUseCase


def create_app(
    ingest_use_case: IngestPdfUseCase, ask_question_use_case: AskQuestionUseCase
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        ingest_use_case: Use case for PDF ingestion.
        ask_question_use_case: Use case for question answering.
    
    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="RAG API",
        description="Retrieval Augmented Generation system with PDF support",
        version="1.0.0",
    )

    @app.post("/upload")
    async def upload_pdf(file: UploadFile = File(...)):
        """Upload a PDF file for processing.
        
        Args:
            file: PDF file to upload.
        
        Returns:
            Document ID and filename.
        """
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        document_id = ingest_use_case.execute(file_path)
        return {"document_id": document_id, "filename": file.filename}

    @app.post("/ask")
    async def ask_question(question: Question):
        """Ask a question based on uploaded documents.
        
        Args:
            question: The question to answer.
        
        Returns:
            Answer with source citations and confidence scores.
        """
        answer = ask_question_use_case.execute(question)
        return {
            "answer": answer.text,
            "citations": [
                {
                    "page_number": chunk.page_number,
                    "document_name": chunk.filename,
                    "score": chunk.score,
                    "text": chunk.text[:100],
                }
                for chunk in answer.used_chunks
            ],
        }

    return app

