"""PDF parsing implementation using PyMuPDF."""

import pymupdf

from src.rag_service.domain.models import Page
from src.rag_service.domain.ports import PdfParser


class PymupdfParser(PdfParser):
    """PDF parser using the PyMuPDF library.
    
    Extracts text content from PDF files page by page.
    """

    def parse(self, file_path: str) -> list[Page]:
        """Extract pages from a PDF file.
        
        Args:
            file_path: Path to the PDF file to parse.
        
        Returns:
            List of pages with extracted text content.
        
        Raises:
            ValueError: If PDF parsing fails.
        """
        pages = []

        try:
            with pymupdf.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages(), start=1):
                    text = page.get_text()
                    if text.strip():
                        pages.append(Page(number=page_num, content=text))
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file: {file_path}") from e

        return pages

