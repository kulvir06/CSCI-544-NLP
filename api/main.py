from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError
import uvicorn
from api import config
from corpus.pipeline import ingest_upload
from corpus.schema import DocumentMetadata, DocumentStatus, IngestionResult

app = FastAPI()

@app.get("/")
def root():
    return "hello"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "network_ingestion_enabled": config.NETWORK_INGESTION_ENABLED,
        "supported_upload_types": sorted(config.SUPPORTED_UPLOAD_SUFFIXES),
    }

class UserQuery(BaseModel):
    query: str


@app.post("/documents/upload", response_model=IngestionResult)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    issuer: Optional[str] = Form(default=None),
    source_url: Optional[str] = Form(default=None),
    published_at: Optional[date] = Form(default=None),
    effective_from: Optional[date] = Form(default=None),
    effective_to: Optional[date] = Form(default=None),
    rule_family: Optional[str] = Form(default=None),
    status: DocumentStatus = Form(default=DocumentStatus.UNKNOWN),
):
    content = await file.read(config.MAX_UPLOAD_BYTES + 1)
    try:
        metadata = DocumentMetadata(
            original_filename=file.filename or "uploaded-document",
            title=title,
            issuer=issuer,
            source_url=source_url,
            published_at=published_at,
            effective_from=effective_from,
            effective_to=effective_to,
            rule_family=rule_family,
            status=status,
        )
        return ingest_upload(metadata.original_filename, content, metadata)
    except (ValueError, ValidationError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.post("/query/")
async def user_query(q: UserQuery):
    # Keep optional LLM/vector dependencies out of the upload-only corpus path.
    from api import llm, orchestrator

    res = orchestrator.get_context(q.query)
    enhanced_query = orchestrator.enhance_user_query(
        query = q.query, 
        context = res["context"],
        sources = res["sources"]
    )
    ollama_client = llm.create_ollama_client()
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
