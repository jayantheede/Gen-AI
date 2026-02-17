from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from .chat_engine import ChatEngine

app = FastAPI(
    title="Multi-Mode RAG API",
    description="Interior design RAG system with Standard, Corrective, and Speculative modes",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatEngine
engine = ChatEngine()

class QuestionRequest(BaseModel):
    question: str
    rag_mode: str = "standard"

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "initialized"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    print(f"MAIN_API: Received question: {request.question}, mode: {request.rag_mode}")
    try:
        response = engine.ask(request.question, rag_mode=request.rag_mode)
        return response
    except Exception as e:
        print(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Static Files Mounts
if os.path.exists("Data"):
    app.mount("/data", StaticFiles(directory="Data"), name="data")
    print("[OK] Mounted /data directory")
    
    if os.path.exists("Data/processed/images"):
        app.mount("/images", StaticFiles(directory="Data/processed/images"), name="images")
        print("[OK] Mounted /images directory")

# Mount Frontend - MUST BE LAST
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
    print("[OK] Mounted PREMIUM FRONTEND at /")
else:
    @app.get("/")
    async def root():
        return {"status": "Backend Active", "info": "Frontend files missing"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)