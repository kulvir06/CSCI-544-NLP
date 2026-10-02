from dataclasses import dataclass
from pydantic import BaseModel, Field


# metadata schema
@dataclass
class ChunkMetadata:
    producer: str
    creator: str
    creation_date: str
    source: str = Field(description="original PDF file name")
    total_pages: int = Field(description="total number of pages")
    page_label: int = Field(description="page index in the document")
    chunk_index: int = Field(description="chunk index in this document")
    title: str = Field(description="optional: title of the document")
    # keywords: list = Field(description="optional: list of keywords")


# chunk-metadata
class Chunk(BaseModel):
    document_chunk: str = Field(description="chunks of the document")
    chunk_metadata: ChunkMetadata
