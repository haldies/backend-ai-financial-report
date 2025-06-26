import csv
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage


def create_vector_index_from_qa_csv(csv_path, vector_store, embed_model, persist_dir="storage"):
    try:
        print(f"📂 Membaca data dari file CSV: {csv_path}")

        all_nodes = []

        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pertanyaan = row.get("Pertanyaan", "").strip()
                jawaban = row.get("Jawaban", "").strip()
                bank = row.get("Bank", "").strip()
                tahun = row.get("Tahun", "").strip()

                text = jawaban

                metadata = {
                    "pertanyaan": pertanyaan,
                    "bank": bank,
                    "tahun": tahun,
                }

                node = TextNode(text=text, metadata=metadata)
                all_nodes.append(node)

        print(f"📚 Total Q&A yang dimuat: {len(all_nodes)}")

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir
        )

        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )

        index.storage_context.persist()

        print("✅ Index berhasil dibuat dan disimpan ke Qdrant + local storage")
        return index

    except Exception as e:
        print(f"❌ Gagal membuat index dari CSV: {e}")
        return None


def load_index(vector_store, embed_model, persist_dir="storage"):
    try:
        print("🔄 Memuat index dari storage...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model
        )
        print("✅ Index berhasil dimuat dari Qdrant dan storage")
        return index

    except Exception as e:
        print(f"❌ Gagal memuat index: {e}")
        return None
