import pymupdf
from rag_service.domain.ports import PdfParser
from rag_service.domain.models import Document, Page, Chunk

class PymupdfParser(PdfParser):
    def parse(self, file_path: str) -> list[Page]:
        pages = []

        try:
            with pymupdf.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages(), start=1):
                    text = page.get_text()
                    if text.strip():
                        pages.append(Page(number=page_num, content=text))
        except:
            raise ValueError(f"Failed to parse PDF file: {file_path}")

        return pages
    
class PdfPlumberParser(PdfParser):
    def parse(self, file_path: str) -> list[Page]:
        pages = []
        return pages