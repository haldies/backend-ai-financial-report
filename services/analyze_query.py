import json
import re
from llama_index.core.llms import ChatMessage
from services.model_init import llm 
from services.searcher import similarity_search_dual

def normalize_metadata_filter(filter_dict):
    if not filter_dict:
        return {}
    return {
        key: value.upper() if isinstance(value, str) else value
        for key, value in filter_dict.items()
    }

def analyze_query_with_llm(query_user: str):
    prompt = f"""
    Kamu adalah asisten cerdas yang menganalisis pertanyaan dari user untuk kebutuhan pencarian laporan keuangan bank.

    Tugasmu:
    1. Tentukan apakah pertanyaan ini membutuhkan 1 query atau perbandingan (2 query).
    2. Jika 1 query: keluarkan satu query, dan metadata filter (bank, tahun, jenis laporan jika ada).
    3. Jika 2 query: ekstrak dua pertanyaan dengan perbedaan tahun (atau bank), dan filter metadata masing-masing.

    Contoh output format JSON:
    {{
      "num_queries": 2,
      "query1": "Berapa EPS Bank BCA tahun 2023?",
      "query2": "Berapa EPS Bank BCA tahun 2024?", 
      "filter1": {{"bank": "PT BANK CENTRAL ASIA TBK", "tahun": "2023"}},
      "filter2": {{"bank": "PT BANK CENTRAL ASIA TBK", "tahun": "2024"}}
    }}

    Hanya balas dalam format JSON valid tanpa penjelasan atau komentar.
    Pertanyaan user:
    "{query_user}"
    """

    try:
        response = llm.chat(messages=[ChatMessage(role="user", content=prompt)])
        response_text = response.message.content.strip()

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("Tidak ditemukan blok JSON dalam respons.")

        json_block = match.group(0)
        result = json.loads(json_block)

        if "filter1" in result:
            result["filter1"] = normalize_metadata_filter(result["filter1"])
        if "filter2" in result:
            result["filter2"] = normalize_metadata_filter(result["filter2"])

        return result

    except Exception as e:
        print(f"❌ Gagal parsing output LLM: {e}")
        print("📄 Response LLM:\n", response_text)
        return None

def smart_rag_search(index, user_query, similarity_top_k=3):
    analysis = analyze_query_with_llm(user_query)
    print("🔍 Analisis query:", json.dumps(analysis, indent=2))

    if not analysis:
        print("❌ Gagal menganalisis query.")
        return [], [], "Gagal menganalisis pertanyaan."

    # Jika sapaan atau bukan pertanyaan data
    if analysis.get("num_queries", 0) == 0:
        message = analysis.get("message", "Silakan ajukan pertanyaan seputar laporan keuangan.")
        print("💬 Jawaban langsung dari LLM:", message)
        return [], [], message

    if analysis["num_queries"] == 2:
        nodes1, nodes2 = similarity_search_dual(
            index=index,
            query1=analysis["query1"],
            query2=analysis["query2"],
            filter1=analysis.get("filter1"),
            filter2=analysis.get("filter2"),
            similarity_top_k=similarity_top_k
        )
        return nodes1, nodes2, None
    else:
        nodes1, _ = similarity_search_dual(
            index=index,
            query1=analysis["query1"],
            filter1=analysis.get("filter1"),
            similarity_top_k=similarity_top_k
        )
        return nodes1, [], None