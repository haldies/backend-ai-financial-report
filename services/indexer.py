import os
import csv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from services.model_init import embed_model

def create_vector_index(documents, vector_store, embed_model, persist_dir="storage"):
    try:
        print("=" * 50)
        print(f"🔄 [1] Mulai proses chunking dari {len(documents)} dokumen...")
        
        all_nodes = []
        node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=50, include_metadata=True)

        for i, doc in enumerate(documents):
            print(f"📄 Dokumen ke-{i+1}: panjang teks = {len(doc.text)}")
            print(f"📌 Cuplikan teks: {repr(doc.text[:100])}")
            print(f"📄 [1.{i+1}] Memproses dokumen ke-{i+1} dengan metadata: {doc.metadata}")
            nodes = node_parser.get_nodes_from_documents([doc])
            print(f"   └─ Jumlah potongan dari dokumen ini: {len(nodes)}")

            for node in nodes:
                node.metadata.update(doc.metadata)
                if not node.text.strip():
                    print("⚠️  [Kosong] Node dengan teks kosong ditemukan!")

            all_nodes.extend(nodes)

        print(f"📚 [2] Total potongan (chunk) yang dihasilkan: {len(all_nodes)}")

        texts = [node.text for node in all_nodes]
        empty_texts = [i for i, t in enumerate(texts) if not t.strip()]
        if empty_texts:
            print(f"⚠️ [3] Ditemukan {len(empty_texts)} teks kosong pada indeks: {empty_texts}")
        
        print(f"🔠 [4] Menghasilkan embedding untuk {len(texts)} teks...")
        embeddings = embed_model.get_text_embedding_batch(texts)
        print("✅ [5] Embedding selesai")

        output_csv = "chunked_with_embedding.csv"
        print(f"💾 [6] Menyimpan hasil chunk dan embedding ke {output_csv}...")
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["chunk_id", "char_length", "text", "embedding", "bank", "tahun", "jenis_laporan"])
            for i, (node, emb) in enumerate(zip(all_nodes, embeddings)):
                embedding_str = "[" + ",".join([f"{x:.6f}" for x in emb]) + "]"
                writer.writerow([
                    i + 1,
                    len(node.text),
                    node.text,
                    embedding_str,
                    node.metadata.get("bank", ""),
                    node.metadata.get("tahun", ""),
                    node.metadata.get("jenis_laporan", "")
                ])
        print("✅ [7] CSV selesai ditulis")

        print(f"📦 [8] Menyimpan index ke direktori: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        index.storage_context.persist()
        print("✅ [9] Vector index berhasil dibuat dan disimpan ke Qdrant")
        print("=" * 50)

        return index

    except Exception as e:
        print("=" * 50)
        print(f"❌ [ERROR] Gagal membuat vector index: {e}")
        print("=" * 50)
        return None

def load_index(vector_store):
    try:
        print("🔄 [LOAD] Memuat index dari storage...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="storage"
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model
        )
        print("✅ [LOAD] Index berhasil dimuat")
        return index
    except Exception as e:
        print(f"❌ [LOAD ERROR] Error loading index: {e}")
        return None
