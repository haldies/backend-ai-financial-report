from llama_index.core.llms import ChatMessage
from services.model_init import llm 
from llama_index.core.schema import NodeWithScore

def generate_answer_with_llm(query: str, contexts: list[NodeWithScore]) -> str:
    context_texts = [node.node.text for node in contexts]
    context_combined = "\n\n".join(context_texts[:5]) 

    messages = [
        ChatMessage(
            role="system",
            content="Kamu adalah asisten keuangan profesional. Jawablah pertanyaan berdasarkan dokumen laporan keuangan yang tersedia. Jika tidak ada informasi yang relevan, jawab: 'Maaf, informasi tersebut tidak tersedia dalam dokumen.'"
        ),
        ChatMessage(
            role="user",
            content=f"Pertanyaan: {query}\n\n=== Konteks Dokumen ===\n{context_combined}"
        )
    ]

    try:
        response = llm.chat(messages)
        return response.message.content.strip()
    except Exception as e:
        print("❌ Error:", e)
        return f"❌ Gagal menghasilkan jawaban dari LLM: {str(e)}"
