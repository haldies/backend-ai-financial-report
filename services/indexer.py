import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from services.model_init import embed_model
def create_vector_index(documents, vector_store, embed_model, persist_dir="storage"):
    try:
        print(f"🔄 Mulai proses chunking dari {len(documents)} dokumen...")

        node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"📚 Total potongan (chunk) yang dihasilkan: {len(nodes)}")

        texts = [node.text for node in nodes]
        embeddings = embed_model.get_text_embedding_batch(texts)
        print("✅ Embedding selesai")

        os.makedirs(persist_dir, exist_ok=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        index.storage_context.persist()
        print("✅ Vector index berhasil dibuat dan disimpan ke Qdrant")

        return index

    except Exception as e:
        print(f"❌ Gagal membuat vector index: {e}")
        return None

def load_index(vector_store):
    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="storage"
        )
        print("🔄 Memuat index dari storage...")
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model 
        )
        return index 
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return None
