from vector import db, user
from api import constants


def enhance_user_query(query: str, context: str, sources: list):
    enhanced_query = f'''
    You are acting as a legal advisor for the user. The user can be a student or a working professional in or outside the United States of America.
    Your job is to resolve the question that the user has regarding the legality of immigration/legal aspects of visa and other immigration related questions.
    Your response should only be based the context that will be provided to you below as a variable whose name is context.
    If you cannot find a relevant answer in the context provided, your response should be that you could not find any relevant information regarding the question.
    Make sure that if you do not find anything that is relevant to the context, you give a single line response and do not provide any other information on your own
    If you provide any information that is not part of the context, the entire output will be incorrect and will fail the purpose of the project
    Do not ask a follow up question if you do not get an answer from the context.
    Just respond with - i am unable to answer you question at the moment
    The question asked by the user will be provided to you in the variable USER_QUESTION below.
    While answering the question make sure you mention the source of the documents as provided in the prompt in the below variable sources
    Response format: Your response should strictly be of this format, you cannot respond in any other way

    USER_QUESTION = {query}
    
    CONTEXT = {context}
    '''

    return enhanced_query


def get_source_from_metadata(metadatas):
    sources = []
    for metadata in metadatas:
        for md in metadata:
            sources.append(md["source"])
    return sources

def get_context(query:str):
    chroma_client = db.initialise_client()
    res = user.process_query(query, chroma_client, constants.N_RESULTS)
    documents = res["documents"]
    sources = get_source_from_metadata(res["metadatas"])
    return {
        "context": str(documents),
        "sources": sources
    }