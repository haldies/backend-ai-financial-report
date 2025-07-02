from fastapi import FastAPI, UploadFile, File, Query,Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pprint

from llama_index.core.llms import ChatMessage
import shutil
import os
from services.create_vector_index_from_qa_csv import create_vector_index_from_qa_csv


from services.extractor import extract_pdf_with_gemini
from services.indexer import create_vector_index, load_index
from services.searcher import similarity_search_dual
from services.model_init import vector_store, embed_model
from services.generator import generate_answer_with_llm
from services.analyze_query import smart_rag_search


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
    history: Optional[List[dict]] = [] 
    
    
@app.post("/chat")
async def rag_query(payload: ChatRequest = Body(...)):
    query = payload.query
    history_dict = payload.history or []

    index = load_index(vector_store)
    
    if not index:
        return {"error": "❌ Index belum tersedia di Qdrant. Silakan upload dokumen terlebih dahulu."}

    try:
        nodes1, nodes2, message = smart_rag_search(index, query)
    except Exception as e:
        print(f"❌ Error saat smart_rag_search: {e}")
        return {"error": "Terjadi kesalahan saat pencarian."}

    if message:
        print(f"💬 Dikenali sebagai sapaan/non-query: {message}")
        generated_answer_clean = message
    else:
        nodes = (nodes1 or []) + (nodes2 or [])
        if not nodes:
            print("⚠️ Tidak ada hasil dari similarity search.")
            generated_answer_clean = "Maaf, informasi tersebut tidak tersedia dalam dokumen."
        else:
            context_combined = "\n\n".join([n.node.get_content() for n in nodes])

            history = [ChatMessage(role=h.get("role"), content=h.get("content")) for h in history_dict]

            generated_answer_clean, new_history = generate_answer_with_llm(query, context_combined, history)

            new_history_dict = [{"role": msg.role, "content": msg.content} for msg in new_history]

    return {
        "query": query,
        "jawaban": generated_answer_clean,
        "jumlah_konteks_digunakan": len(context_combined.split("\n\n")) if not message and nodes else 0,
        "history": new_history_dict if not message and nodes else history_dict  # kembalikan history terbaru
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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
    node_summaries = []

    try:
        all_nodes = index.vector_store.get_nodes()

        for i, node in enumerate(all_nodes[:10], 1): 
            node_summary = {
                "nomor": i,
                "text_snippet": node.text[:100].replace("\n", " ") + "...",
                "metadata": node.metadata
            }
            node_summaries.append(node_summary)

    except Exception as e:
        return {"error": f"Gagal mengambil node: {str(e)}"}

    return {
        "pesan": "Berhasil upload, ekstrak, dan indexing",
        "jumlah_dokumen": len(documents),
        "jumlah_node": len(node_summaries),
        "node_sampel": node_summaries
    }




@app.get("/search")
async def search_similar_text(
    query1: str = Query(..., description="Pertanyaan pertama (contoh: Berapa EPS tahun 2023?)"),
    query2: Optional[str] = Query(None, description="Pertanyaan kedua (contoh: Berapa EPS tahun 2024?)"),
    bank1: Optional[str] = Query(None, description="Nama lengkap bank, contoh: PT BANK MANDIRI (PERSERO) TBK"),
    bank2: Optional[str] = Query(None, description="Nama lengkap bank, contoh: PT BANK CENTRAL ASIA TBK"),
    tahun1: Optional[str] = Query(None, description="Tahun untuk query 1"),
    tahun2: Optional[str] = Query(None, description="Tahun untuk query 2"),
    top_k: int = Query(5, description="Jumlah hasil teratas untuk setiap query")
):

    print("🔍 Menerima permintaan pencarian...")
    print(f"  ✅ Query 1: {query1}")
    print(f"  ✅ Query 2: {query2}")
    print(f"  ✅ Filter 1: {{'bank': {bank1}, 'tahun': {tahun1}}}")
    print(f"  ✅ Filter 2: {{'bank': {bank2}, 'tahun': {tahun2}}}")
    print(f"  ✅ Top K: {top_k}")

    # Load index
    print("📦 Memuat index dari vector store...")
    index = load_index(vector_store)
    if not index:
        print("❌ Gagal memuat index")
        return JSONResponse(content={"error": "❌ Index belum tersedia di Qdrant. Buat index terlebih dahulu."}, status_code=404)
    print("✅ Index berhasil dimuat")

    # Lakukan similarity search
    try:
        print("🔎 Melakukan similarity search...")
        filter1 = {"bank": bank1, "tahun": tahun1}
        filter2 = {"bank": bank2, "tahun": tahun2}
        nodes1, nodes2 = similarity_search_dual(
            index=index,
            query1=query1,
            query2=query2,
            filter1=filter1,
            filter2=filter2,
            similarity_top_k=top_k
        )
    except Exception as e:
        print(f"❌ Error saat melakukan similarity search: {e}")
        return JSONResponse(content={"error": f"❌ Error saat similarity search: {str(e)}"}, status_code=500)

    def format_nodes(nodes):
        return [
            {
                "score": getattr(node, 'score', None),
                "content": node.get_content()[:300],
                "metadata": node.metadata
            }
            for node in nodes
        ] if nodes else []

    formatted_nodes1 = format_nodes(nodes1)
    formatted_nodes2 = format_nodes(nodes2) if query2 else None

    print("📄 Hasil Query 1:")
    pprint.pprint(formatted_nodes1)

    if formatted_nodes2:
        print("📄 Hasil Query 2:")
        pprint.pprint(formatted_nodes2)

    response_data = {
        "query1_results": formatted_nodes1,
        "query2_results": formatted_nodes2,
    }

    return response_data




@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Simpan file temporer
        temp_file_path = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Proses indexing
        index = create_vector_index_from_qa_csv(
            csv_path=temp_file_path,
            vector_store=vector_store,
            embed_model=embed_model
        )

        if index is None:
            return JSONResponse(status_code=500, content={"message": "Gagal membuat index"})

        return {"message": "✅ CSV berhasil diproses dan diindeks ke Qdrant"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"❌ Error: {str(e)}"})


@app.get("/all-nodes")
async def get_all_nodes(limit: int = 100):
    index = load_index(vector_store)
    if not index:
        return JSONResponse(content={"error": "❌ Index belum tersedia di Qdrant."}, status_code=404)

    try:
        all_nodes = index.vector_store.get_nodes()
        total_nodes = len(all_nodes)

        node_list = [
            {
                "nomor": i + 1,
                "text": node.text,
                "metadata": node.metadata
            }
            for i, node in enumerate(all_nodes[:limit])
        ]

        return {
            "total_node_dalam_index": total_nodes,
            "jumlah_node_ditampilkan": len(node_list),
            "data": node_list
        }

    except Exception as e:
        return JSONResponse(content={"error": f"Gagal mengambil node: {str(e)}"}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Welcome to the Laporan Keuangan API. Use /upload/ to upload PDF files."}