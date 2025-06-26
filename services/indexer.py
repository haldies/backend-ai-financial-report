import os
import csv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from services.model_init import embed_model


def create_vector_index(documents, vector_store, embed_model, persist_dir="storage"):
    try:
        print(f"🔄 Mulai proses chunking dari {len(documents)} dokumen...")

        all_nodes = []
        node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=50)


        for doc in documents:
            nodes = node_parser.get_nodes_from_documents([doc])

            for node in nodes:
                node.metadata.update(doc.metadata)  

            all_nodes.extend(nodes)

        print(f"📚 Total potongan (chunk) yang dihasilkan: {len(all_nodes)}")
        
        texts = [node.text for node in nodes]
        embeddings = embed_model.get_text_embedding_batch(texts)
        print("✅ Embedding selesai")

        with open("chunked_with_embedding.csv", mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["chunk_id", "char_length", "text", "embedding", "bank", "tahun","jenis_laporan"])
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
