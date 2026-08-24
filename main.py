"""
FastAPI backend for the Basis-Selection Agent.

Run locally:
    uvicorn main:app --reload --port 8000

Endpoint:
    POST /analyze
    Body: {"x": [...], "y": [...]}
    Returns: JSON result from agent.analyze()
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from agent import analyze

app = FastAPI(title="Basis-Selection Agent API")

# Allow requests from the GitHub Pages frontend (and localhost for dev).
# Tighten allow_origins to your actual GitHub Pages URL once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    x: List[float] = Field(..., min_items=5, description="Observation x-values")
    y: List[float] = Field(..., min_items=5, description="Observation y-values")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Basis-Selection Agent API is running."}


@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    if len(req.x) != len(req.y):
        raise HTTPException(status_code=400, detail="x and y must have the same length.")
    if len(req.x) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 observations.")

    try:
        result = analyze(req.x, req.y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return result
