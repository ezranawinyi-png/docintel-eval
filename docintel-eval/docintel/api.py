"""FastAPI app.   Run:  uvicorn docintel.api:app --reload
Set DOCINTEL_DOCS_DIR=/path/to/txt/files to serve your own documents instead of the demo corpus."""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .service import Service, docs_from_directory

app = FastAPI(title="DocIntel-Eval", version="0.1.0",
              description="Contract field extraction and cited question answering.")

_dir = os.environ.get("DOCINTEL_DOCS_DIR")
service = Service(docs_from_directory(_dir) if _dir else None)


class ExtractRequest(BaseModel):
    text: str
    method: str = "regex-v2"


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", **service.stats()}


@app.post("/extract")
def extract(req: ExtractRequest):
    try:
        return service.extract(req.text, req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    return service.ask(req.question)
