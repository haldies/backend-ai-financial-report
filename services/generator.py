
from llama_index.core.llms import ChatMessage
from services.model_init import llm 
from typing import List, Optional

def generate_answer_with_llm(
    query: str,
    contexts: str,
    history: List[ChatMessage] = None  # menerima history pesan
) -> (str, List[ChatMessage]):

    if history is None:
        history = []

    # System prompt tetap di awal conversation
    system_message = ChatMessage(
        role="system",
        content=(
            "Kamu adalah asisten keuangan profesional. Jawablah pertanyaan berdasarkan dokumen "
            "laporan keuangan yang tersedia. Jika tidak ada informasi yang relevan, jawab: "
            "'Maaf, informasi tersebut tidak tersedia dalam dokumen.'"
        )
    )
    
    # Buat pesan user dengan konteks
    user_message = ChatMessage(
        role="user",
        content=f"Pertanyaan: {query}\n\n=== Konteks Dokumen ===\n{contexts}"
    )

    # Gabungkan pesan: system + history + user
    messages = [system_message] + history + [user_message]

    print("📄 Mengirim pesan ke LLM dengan history:")
    for msg in messages:
        print(f"Role: {msg.role}")
        print(f"Content:\n{msg.content}\n{'-'*40}")

    try:
        response = llm.chat(messages)
        # Tambahkan pesan user dan assistant ke history baru
        new_history = history + [user_message, response.message]
        return response.message.content.strip(), new_history
    except Exception as e:
        print("❌ Error:", e)
        return f"❌ Gagal menghasilkan jawaban dari LLM: {str(e)}", history
