from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
import uvicorn
from api import config
from corpus.pipeline import ingest_upload
from corpus.review import build_review_queue, submit_review
from corpus.schema import (
    DocumentMetadata,
    DocumentReview,
    DocumentReviewRequest,
    DocumentStatus,
    IngestionResult,
    ReviewQueueItem,
)
from retrieval.schema import RetrievalRequest, RetrievalResponse
from retrieval.search import retrieve
from prototype.pipeline import run_prototype
from prototype.schema import PrototypeRequest, PrototypeResponse
from verification.meaning import verify_meaning
from verification.safety import verify_safety
from verification.schema import (
    MeaningVerificationRequest,
    MeaningVerificationResponse,
    SafetyVerificationRequest,
    SafetyVerificationResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="F-1 Verified Explanation Prototype",
    description=(
        "Local, upload-only prototype for automatic claim-level temporal grounding "
        "and meaning-preservation verification of F-1 employment guidance."
    ),
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/app")


@app.get("/app", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def frontend_styles():
    """Support the dashboard's relative asset path and direct-file previews."""
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def frontend_script():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="text/javascript")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "network_ingestion_enabled": config.NETWORK_INGESTION_ENABLED,
        "supported_upload_types": sorted(config.SUPPORTED_UPLOAD_SUFFIXES),
        "temporal_index_ready": config.RETRIEVAL_INDEX_PATH.exists(),
        "semantic_index_ready": config.SEMANTIC_PASSAGE_INDEX_PATH.exists(),
        "temporal_model": config.OLLAMA_TEMPORAL_MODEL,
        "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
    }

@app.post("/documents/upload", response_model=IngestionResult)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    issuer: Optional[str] = Form(default=None),
    source_url: Optional[str] = Form(default=None),
    published_at: Optional[date] = Form(default=None),
    effective_from: Optional[date] = Form(default=None),
    effective_to: Optional[date] = Form(default=None),
    valid_as_of: Optional[date] = Form(default=None),
    retrieved_at: Optional[date] = Form(default=None),
    date_evidence: Optional[str] = Form(default=None),
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
            valid_as_of=valid_as_of,
            retrieved_at=retrieved_at,
            date_evidence=date_evidence,
            rule_family=rule_family,
            status=status,
        )
        return ingest_upload(metadata.original_filename, content, metadata)
    except (ValueError, ValidationError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/documents/review-queue", response_model=list[ReviewQueueItem])
def review_queue(include_resolved: bool = False):
    return build_review_queue(include_resolved=include_resolved)


@app.post("/documents/{document_id}/review", response_model=DocumentReview)
def review_document(document_id: str, request: DocumentReviewRequest):
    try:
        return submit_review(document_id, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_passages(request: RetrievalRequest):
    return retrieve(request)


@app.post("/verify/meaning", response_model=MeaningVerificationResponse)
def verify_explanation_meaning(request: MeaningVerificationRequest):
    return verify_meaning(request)


@app.post("/verify/safety", response_model=SafetyVerificationResponse)
def verify_explanation_safety(request: SafetyVerificationRequest):
    return verify_safety(request)


@app.post("/prototype/query", response_model=PrototypeResponse)
def prototype_query(request: PrototypeRequest):
    return run_prototype(request)

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port = 8000,
        reload = True
    )
