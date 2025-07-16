from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from app.config import PINECONE_API_KEY
from app.rag_utils import (
    extract_text_from_pdf,
    split_text,
    store_chunks,
    ask_question,
    initialize,
    index  # This is for direct Pinecone access
)
from app.database import SessionLocal
from app.models import ChunkMetadata

app = FastAPI()  # Initialize FastAPI App
UPLOAD_DIR = "uploads" # ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup") # Startup and model initializaiton
def on_startup():
    print("FastAPI startup: calling initialize()")
    initialize(pinecone_api_key=PINECONE_API_KEY)

@app.post("/upload/") #To upload pdf file in endpoint
async def upload_file(file: UploadFile):
    print("/upload/ endpoint hit")

    contents = await file.read()
    print(f"Received file: {file.filename} ({len(contents)} bytes)")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    print(f"File saved to {file_path}")

    text = extract_text_from_pdf(contents)
    print(f" Extracted text length: {len(text)}")

    chunks = split_text(text)
    print(f"Split into {len(chunks)} chunks")

    count = store_chunks(chunks, filename=file.filename)
    print(f"Stored {count} chunks")

    return {
        "filename": file.filename,
        "text_length": len(text),
        "chunks_stored": count
    }

@app.post("/ask/") #Ask question to endpoint
async def question_answer(question: str = Form(...)):
    print(" /ask/ endpoint hit")
    print(" Question received:", question)

    answer, sources = ask_question(question)
    print(" Answer generated")

    return JSONResponse(content={
        "question": question,
        "answer": answer,
        "sources": sources
    })


