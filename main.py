from fastapi import FastAPI, UploadFile, File, Query,Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os


from services.extractor import extract_pdf_with_gemini
from services.indexer import create_vector_index, load_index
from services.searcher import similarity_search
from services.model_init import vector_store, embed_model
from services.generator import generate_answer_with_llm


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def rag_query(payload: ChatRequest = Body(...)):
    query = payload.query
    index = load_index(vector_store)
    if not index:
        return {"error": "❌ Index belum tersedia di Qdrant. Silakan upload dokumen terlebih dahulu."}

    results = similarity_search(index, query, similarity_top_k=3)
    if not results:
        return {"error": "❌ Tidak ada dokumen yang relevan ditemukan."}

    answer = generate_answer_with_llm(query, results)

    return {
        "query": query,
        "jawaban": answer,
        "jumlah_konteks_digunakan": len(results),
        "konteks": [
            {
                "score": round(node.score, 4),
                "text": node.node.text[:1000] + "..." if len(node.node.text) > 1000 else node.node.text
            }
            for node in results
        ]
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Simpan file PDF yang diupload
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    documents = extract_pdf_with_gemini(file_location)
    if not documents:
        return {"error": "Gagal mengekstrak QnA dari PDF"}

    index = create_vector_index(documents, vector_store, embed_model)
    if not index:
        return {"error": "Gagal membuat vector index ke Qdrant"}

    return {
        "pesan": "Berhasil upload, ekstrak, dan indexing",
        "jumlah_dokumen": len(documents)
    }

@app.get("/search")
async def search_similar_text(query: str = Query(...), top_k: int = 3):
    """Search for similar chunks without generating answer"""
    index = load_index(vector_store)
    if not index:
        return {"error": "❌ Index belum tersedia di Qdrant. Buat index terlebih dahulu."}

    results = similarity_search(index, query, similarity_top_k=top_k)
    
    return {
        "query": query,
        "jumlah_dokumen_ditemukan": len(results),
        "hasil": [
            {
                "score": round(node.score, 4),
                "text": node.node.text[:1000] + "..." if len(node.node.text) > 1000 else node.node.text
            }
            for node in results
        ]
    }
    
    

@app.get("/")
async def root():
    return {"message": "Welcome to the Laporan Keuangan API. Use /upload/ to upload PDF files."}