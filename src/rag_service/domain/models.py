from dataclasses import dataclass
from typing import List
import uuid

@dataclass
class Page:
    number: int
    content: str

@dataclass
class Chunk:
    id: str
    document_id: str
    page_number: int
    text: str

@dataclass
class Document:
    id: str
    filename: str
    pages: List[Page]
    chunks: List[Chunk]

    @staticmethod
    def from_file(file_path: str, pages_text:List[Page]) -> Document:
        return Document(id=str(uuid.uuid4()), 
                        filename=file_path, 
                        pages=pages_text, 
                        chunks=[]
                        )
    
    def add_chunks(self, chunks: List[Chunk]):
        self.chunks = chunks

@dataclass
class Question:
    text: str

@dataclass
class Answer:
    text: str
    used_chunks: List[Chunk]

    
    