from dataclasses import asdict
from importlib import metadata
import chromadb
from vector import constants


def initialise_client():
    return chromadb.PersistentClient(path = constants.DB_PATH)


def add_vectors(collection: chromadb.Collection, vector, document, metadata, id):

    metadata = asdict(metadata) # convert metadata to dict -> collections can only read dict type metadata
    
    collection.add(
        ids = id,
        documents = [document],
        embeddings = [vector],
        metadatas = [metadata],
    )
