"""
Thin FastAPI layer exposing the agent as a service -- this is the
"deployed interface" piece (not a notebook) the TELUS role asks for.

Run: uvicorn src.main:app --reload --port 8000
Then: curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
      -d '{"question": "What is the elevation at 69.3N, -138.9W?"}'
"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent
from tools import list_datasets

app = FastAPI(title="Arctic GeoAgent", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    image_base64: Optional[str] = None


@app.get("/datasets")
def get_datasets():
    """Lists available datasets -- useful for a frontend to show what can be asked."""
    return list_datasets()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = run_agent(req.question, verbose=False)
    return AskResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
