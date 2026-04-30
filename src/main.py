from rag_service.use_cases.ingest_pdf_use_case import IngestPdfUseCase
from rag_service.use_cases.ask_question_use_case import AskQuestionUseCase

from rag_service.infrastructure.pdf_parser import PymupdfParser
from rag_service.infrastructure.simple_chunker import SimpleChunker
from rag_service.infrastructure.embedding import FakeEmbedder
from rag_service.infrastructure.memory_document_repo import MemoryDocumentRepository
from rag_service.infrastructure.memory_vector_repo import MemoryVectorRepository
from rag_service.infrastructure.fake_llm_service import FakeLLMService

from rag_service.domain.models import Question

def main():
    # Initialize infrastructure components
    pdf_parser = PymupdfParser()
    chunker = SimpleChunker()
    embedder = FakeEmbedder()
    document_repo = MemoryDocumentRepository()
    vector_repo = MemoryVectorRepository()
    llm_service = FakeLLMService()

    # Create use cases
    ingest_use_case = IngestPdfUseCase(pdf_parser, chunker, embedder, document_repo, vector_repo)
    ask_question_use_case = AskQuestionUseCase(embedder, vector_repo, llm_service)

    # Ingest a PDF (simulated)
    document_id = ingest_use_case.execute("sample.pdf")
    print(f"Ingested document with ID: {document_id}")

    # Ask a question
    question_text = "What is the content of page 2?"
    question = Question(text=question_text)
    answer = ask_question_use_case.execute(question)

    print(f"Answer: {answer.text}")
    print("Used chunks:")
    for chunk in answer.used_chunks:
        print(f"- Page {chunk.page_number}: {chunk.text}")

if __name__ == "__main__":
    main()