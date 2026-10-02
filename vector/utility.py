from unittest import result
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from vector import schema
from vector import constants

def get_pdf_files(directory: str, extension: str = "*.pdf") -> list[Path]:
    folder_path = Path(directory)
    file_list = list(folder_path.glob(extension))
    return file_list

# split a document into chunks
def get_chunks(file_path: str) -> list:
    pdf_loader = PyPDFLoader(file_path)
    doc = pdf_loader.load()

    # Notes:
    # RecursiveCharacterTextSplitter is a method to read the document and split it(end of page/end of sentence)
    # tiktoken encoder is the why the pages are measured, how to split the content, by characters or by tokens
    # llms are token readable not character readable hence we need tiktoken encoder
    # tiktoken encoding - OpenAI

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name = constants.ENCODING_NAME,
        chunk_size = constants.CHUNK_SIZE,
        chunk_overlap = constants.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(doc)

    pydantic_chunks = []

    for i, chunk in enumerate(chunks):
        chunk_metadata = schema.ChunkMetadata(
            producer = chunk.metadata.get(constants.PRODUCER, "-"),
            creator = chunk.metadata.get(constants.CREATOR, "-"),
            creation_date = chunk.metadata.get(constants.CREATION_DATE, "-"),
            source = chunk.metadata.get(constants.SOURCE, "-"),
            total_pages = chunk.metadata.get(constants.TOTAL_PAGES, -1),
            page_label = chunk.metadata.get(constants.PAGE_LABEL, 0),
            chunk_index = i,
            title = chunk.metadata.get(constants.SOURCE, "-").split("/")[-1], # to be improved
            # keywords = None
        )

        pydantic_chunk = schema.Chunk(
            document_chunk = chunk.page_content,
            chunk_metadata = chunk_metadata
        )

        pydantic_chunks.append(pydantic_chunk)

    return pydantic_chunks

# get vectors for a chunk
def get_vectors(embedding_model, chunk):
    vector = embedding_model.embed_query(chunk)
    return vector

# get vector embeddings for the documents
def get_document_embeddings(data, embedding_model) -> list:
    document_embeddings = []
    for doc in data:
        temp = []
        for chunks in doc:
            v = get_vectors(embedding_model, chunks.document_chunk)
            temp.append({
                "document": chunks.document_chunk,
                "vectors": v,
                "metadata": chunks.chunk_metadata
            })

        document_embeddings.append(temp)
    
    return document_embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = constants.EMBEDDING_MODEL_NAME
    )

    return embedding_model

