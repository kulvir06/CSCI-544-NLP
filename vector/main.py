import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
import urllib3
from vector import utility
from vector import constants
from vector import db

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Collections are where you'll store your embeddings, documents, and any additional metadata.
# Collections index your embeddings and documents, and enable efficient retrieval and filtering. 

pdf_files = utility.get_pdf_files(constants.DOCUMENT_CORPUS_DIRECTORY) # get pdf files list

# store chunks of all documents in-memory 
# TODO -> improve this by making it persistent as a transaction
data = []
for f in pdf_files:
    data.append(utility.get_chunks(f))

# structure of variable -> data :
# list[[(class Chunk), (), ()......]]
# class Chunk (document_chunk, chunk_metadata)
# data = [[C1 of doc1, C2 of doc1, C3 of doc1], [C1 of doc2, C2 of doc2, C3 of doc2]]


# embedding model to create vectors 
embedding_model = utility.get_embedding_model()

document_embeddings = utility.get_document_embeddings(
    data = data,
    embedding_model = embedding_model
)

# this is a local chroma client -> can be extended to chroma cloud
chroma_client = db.initialise_client()

collection = chroma_client.get_or_create_collection(name = constants.COLLECTION_NAME)

for document_embedding in document_embeddings:
    for chunks in document_embedding:
        db.add_vectors(
            id = str(collection.count()+1),
            collection = collection, 
            vector = chunks["vectors"], 
            document = chunks["document"],
            metadata = chunks["metadata"]
        )
