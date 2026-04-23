from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="genai-rag-service")

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest) -> dict:
    # Replace stubs with real retrieval + generation integration
    return {
        "question": req.question,
        "answer": "stubbed-response",
        "sources": [],
    }
