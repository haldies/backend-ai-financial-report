from services.qdrant_init import vector_store, qdrant_client
exists = qdrant_client.collection_exists(collection_name="laporan-keuangan")
print("Collection masih ada?" , exists)
# qdrant_client.delete_collection(collection_name="laporan-keuangan")
# collections = qdrant_client.get_collections()
# print(collections)
# exists = qdrant_client.collection_exists(collection_name="laporan-keuangan")
# print("Collection masih ada?" , exists)
# Setelah index dibuat dan data dimasukkan


collection_name = "laporan-keuangan"
try:
    qdrant_client.create_payload_index(collection_name, "bank", "keyword")
    qdrant_client.create_payload_index(collection_name, "tahun", "keyword")
    print("✅ Berhasil membuat index untuk metadata.")
except Exception as e:
    print("⚠️ Gagal membuat index metadata:", e)

collection_info = qdrant_client.get_collection(collection_name=collection_name)
print("📋 Daftar field index yang tersedia:")
print(collection_info.payload_schema)
