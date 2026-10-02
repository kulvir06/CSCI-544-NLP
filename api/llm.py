import ollama
from ollama import AsyncClient, ChatResponse
from api import constants

def create_ollama_client():
    client = AsyncClient()
    return client

async def prompt_response(prompt:str, client: AsyncClient):
    response = await client.generate(
        model = constants.OLLAMA_MODEL,
        prompt = prompt,
    )
    return response
    
