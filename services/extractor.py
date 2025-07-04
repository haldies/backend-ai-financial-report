import os
import base64
import csv
import re
from PyPDF2 import PdfReader, PdfWriter
from services.model_init import gemini_client
from tempfile import NamedTemporaryFile
from llama_index.core.schema import Document

def extract_bank_and_year_from_question(question: str):
    bank_match = re.search(
        r"((?:PT\.?\s*)?BANK[\w\s().,&'-]*)\s*(?:pada|tahun|[.,?]|$)", 
        question, 
        re.IGNORECASE
    )
    tahun_match = re.search(r"tahun\s+(\d{4})", question, re.IGNORECASE)
    tahun = tahun_match.group(1) if tahun_match else "0000"
    
    if bank_match:
        bank_text = bank_match.group(1).lower()
        if "mandiri" in bank_text:
            bank = "PT BANK MANDIRI (PERSERO) TBK"
        elif "bca" in bank_text or "central asia" in bank_text:
            bank = "PT BANK CENTRAL ASIA TBK"
        else:
            bank = bank_match.group(1).upper().strip()
    else:
        bank = "BANK TIDAK DIKETAHUI"
    
    return bank, tahun



def extract_pdf_with_gemini(pdf_path, output_path="output_qna.csv"):
    reader = PdfReader(pdf_path)
    all_qna = []
    documents = []

    for i, page in enumerate(reader.pages):
        print(f"🔍 Memproses halaman {i + 1}...")

        # Simpan halaman sebagai file PDF sementara
        writer = PdfWriter()
        writer.add_page(page)

        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            writer.write(temp_pdf)
            temp_pdf_path = temp_pdf.name

        with open(temp_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(temp_pdf_path)

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        prompt_text = """
        Anda adalah asisten cerdas yang membantu mengekstrak data dari laporan keuangan perusahaan.

        Tugas Anda:
        1. Baca seluruh isi halaman ini dengan cermat
        2. Ekstrak setiap informasi penting dalam bentuk pasangan pertanyaan dan jawaban (Q&A)
        3. Gunakan satu fakta untuk satu Q&A. Jangan menggabungkan beberapa informasi jadi satu jawaban
        4. Jika ada angka, istilah, entitas, atau nilai spesifik (seperti pendapatan, laba, beban, rasio keuangan), buat satu pertanyaan terpisah untuk masing-masing
        5. Setiap pertanyaan wajib mencantumkan nama lengkap bank dan tahun laporan di dalam teksnya

        Format Wajib:
        Q: [pertanyaan dalam Bahasa Indonesia]
        A: [jawaban lengkap dan jelas berdasarkan isi halaman]

        Jangan menyisipkan opini, ringkasan, atau interpretasi tambahan. Ekstraksi harus lengkap dan akurat sesuai isi halaman.
        """.strip()

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

        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents
            )
            response_text = response.text.strip()
        except Exception as e:
            print(f"❌ Gagal mendapatkan respons Gemini di halaman {i+1}: {e}")
            continue

        with open(f"debug_page_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(response_text)

        
        qna_blocks = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", response_text, re.DOTALL)

        if not qna_blocks:
            print(f"Tidak ditemukan Q&A yang valid di halaman {i+1}")
            continue

        for q, a in qna_blocks:
            try:
                q = q.strip()
                a = a.strip()
                bank, tahun = extract_bank_and_year_from_question(q)

                all_qna.append([q, a, bank, tahun, "Laporan Tidak Diketahui"])

                documents.append(Document(
                    text=f"Q: {q}\nA: {a}",
                    metadata={
                        "bank": bank,
                        "tahun": tahun,
                        "jenis_laporan": "Laporan Tidak Diketahui",
                        "page": i + 1,
                        "source": os.path.basename(pdf_path)
                    }
                ))
            except Exception as parse_err:
                print(f" Gagal parsing QnA di halaman {i + 1}: {parse_err}")
                continue

    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pertanyaan", "Jawaban", "Bank", "Tahun", "Jenis Laporan"])
        writer.writerows(all_qna)

    print(f"✅ Semua Q&A disimpan di: {output_path}")
    print(f"📦 Total dokumen yang dihasilkan: {len(documents)}")

    return documents
