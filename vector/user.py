from vector import utility
from vector import db
from vector import constants

def process_query(query: str, chroma_client, n_results):
    collection = chroma_client.get_collection(name = constants.COLLECTION_NAME)

    query_embedding = utility.get_vectors(
        embedding_model = utility.get_embedding_model(),
        chunk = query
    )

    res = collection.query(
        query_embeddings = query_embedding,
        n_results = n_results
    )

    return res
