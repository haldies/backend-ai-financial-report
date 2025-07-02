
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

    system_message = ChatMessage(
        role="system",
        content=(
            "Kamu adalah asisten keuangan profesional yang membantu pengguna memahami laporan keuangan."
            " Saat menjawab, jangan hanya menyebutkan angka atau data, tapi juga jelaskan secara singkat "
            "apa arti data tersebut dan mengapa itu penting, terutama jika mengandung istilah teknis seperti EPS, ROE, "
            "liabilitas, atau EBITDA. Gunakan bahasa yang mudah dimengerti oleh orang awam."
            "Jika informasi tidak tersedia di dokumen, jawab: 'Maaf, informasi tersebut tidak tersedia dalam dokumen.'"
        )
    )

    
    user_message = ChatMessage(
        role="user",
        content=f"Pertanyaan: {query}\n\n=== Konteks Dokumen ===\n{contexts}"
    )

    messages = [system_message] + history + [user_message]

    print("📄 Mengirim pesan ke LLM dengan history:")
    for msg in messages:
        print(f"Role: {msg.role}")
        print(f"Content:\n{msg.content}\n{'-'*40}")

    try:
        response = llm.chat(messages)
        new_history = history + [user_message, response.message]
        return response.message.content.strip(), new_history
    except Exception as e:
        print("❌ Error:", e)
        return f"❌ Gagal menghasilkan jawaban dari LLM: {str(e)}", history
