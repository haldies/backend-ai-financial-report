
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
            "Kamu adalah asisten keuangan profesional yang membantu pengguna memahami laporan keuangan. "
            "Saat menjawab, prioritaskan informasi yang tersedia dalam riwayat percakapan sebelum melihat konteks dokumen. "
            "Jangan hanya menyebutkan angka atau data, tapi juga jelaskan secara singkat arti dan pentingnya. "
            "Jika informasi tidak tersedia baik di history maupun konteks, jawab: 'Maaf, informasi tersebut tidak tersedia dalam dokumen.'"
        )
    )

    user_message = ChatMessage(
        role="user",
        content=(
            f"Pertanyaan: {query}\n\n"
            "Jawab berdasarkan informasi yang tersedia di riwayat percakapan (jika relevan), "
            "sebelum memeriksa konteks dokumen. Hanya gunakan konteks dokumen jika informasi belum ada di history.\n\n"
            "=== Konteks Dokumen ===\n"
            f"{contexts}"
        )
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
