import os
import base64
from PyPDF2 import PdfReader, PdfWriter
from services.model_init import gemini_client
from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
from llama_index.core.schema import Document

def extract_pdf_with_gemini(pdf_path):
    reader = PdfReader(pdf_path)
    all_qna_docs = []

    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            writer.write(temp_pdf)
            temp_pdf_path = temp_pdf.name

        with open(temp_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(temp_pdf_path)

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        prompt_text = (  """
                Kamu akan membantu mengekstrak informasi penting dari laporan keuangan perusahaan terbuka di Indonesia, dan menyusunnya dalam bentuk pasangan pertanyaan–jawaban untuk digunakan pada sistem Retrieval-Augmented Generation (RAG).

                Aturan Penulisan:
                - Gunakan **bahasa Indonesia formal dan jelas**.
                - Jawaban harus **langsung, padat, dan berdasarkan data aktual** dari halaman yang diberikan.
                - Pertanyaan harus umum namun relevan, seolah ditanyakan oleh pengguna kepada chatbot.

                Format Output:
                Tulis hasil dalam format:
        
                Q: \[Pertanyaan 1]
                A: \[Jawaban 1]

                Q: \[Pertanyaan 2]
                A: \[Jawaban 2]

                Contoh :

                Q: Berapa total aset PT Bank ABC pada tahun 2024?
                A: Total aset perusahaan pada tahun 2024 mencapai Rp1.449 triliun, meningkat dari Rp1.408 triliun di tahun 2023.

                Q: Bagaimana tren laba bersih perusahaan?
                A: Laba bersih tahun 2024 meningkat 15% dibandingkan tahun 2023, mencapai Rp45 triliun.

                Q: Apa fokus strategi perusahaan menurut Direksi?
                A: Manajemen menyatakan bahwa fokus perusahaan adalah pada digitalisasi layanan dan efisiensi operasional.

                ```

                Sekarang, ekstrak semua isi halaman dengan lengkap PDF berikut menjadi kumpulan pertanyaan dan jawaban seperti format di atas.
                """
                )
        prompt_text = prompt_text.strip()

        contents = [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_base64
                        }
                    }
                ]
            }
        ]

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )

        qna_pairs = response.text.strip().split("\n\n")
        for pair in qna_pairs:
            if pair.strip():
                all_qna_docs.append(Document(text=pair.strip()))

    return all_qna_docs
