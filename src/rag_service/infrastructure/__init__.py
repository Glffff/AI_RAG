"""Infrastructure layer implementations of domain ports.

This module contains concrete implementations of the abstract interfaces defined
in the domain layer:

Modules:
    embedding: Text embedding service using Sentence Transformers.
    llm_service: LLM answer generation using Ollama.
    pdf_parser: PDF file parsing using PyMuPDF.
    simple_chunker: Text chunking with sliding window overlap.
    vector_repo: Vector storage implementations (in-memory and Qdrant).
    memory_document_repo: In-memory document storage.
"""
