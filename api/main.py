from pydantic import BaseModel
from api import constants
from api import orchestrator
from vector import user
from vector import db
from fastapi import FastAPI
import uvicorn
from api import llm

# chroma_client = db.initialise_client()

# print(user.process_query("test query", chroma_client, constants.N_RESULTS))

ollama_client = llm.create_ollama_client()

app = FastAPI()

@app.get("/")
def root():
    return "hello"

class UserQuery(BaseModel):
    query: str

@app.post("/query/")
async def user_query(q: UserQuery):
    res = orchestrator.get_context(q.query)
    enhanced_query = orchestrator.enhance_user_query(
        query = q.query, 
        context = res["context"],
        sources = res["sources"]
    )
    res = await llm.prompt_response(enhanced_query, ollama_client)
    # print(res)
    return res

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port = 8000,
        reload = True
    )