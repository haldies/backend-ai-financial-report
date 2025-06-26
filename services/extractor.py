import os
import base64
import csv
from PyPDF2 import PdfReader, PdfWriter
from services.model_init import gemini_client
from tempfile import NamedTemporaryFile
from llama_index.core.schema import Document
import re


def extract_bank_and_year_from_question(question: str):
    bank_match = re.search(r"PT\s+Bank\s+(.+?)\s+Tbk", question, re.IGNORECASE)
    tahun_match = re.search(r"tahun\s+(\d{4})", question)

    bank = f"PT Bank {bank_match.group(1).strip()} Tbk" if bank_match else "Bank Tidak Diketahui"
    tahun = tahun_match.group(1) if tahun_match else "0000"
    return bank.upper(), tahun

def extract_pdf_with_gemini(pdf_path, output_path="output_qna.csv"):
    reader = PdfReader(pdf_path)
    all_qna = []
    documents = []

    for i, page in enumerate(reader.pages):
        # Simpan halaman ke file PDF sementara
        writer = PdfWriter()
        writer.add_page(page)

        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            writer.write(temp_pdf)
            temp_pdf_path = temp_pdf.name

        # Konversi halaman PDF ke base64
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
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )

        response_text = response.text.strip()

        # Simpan hasil mentah (debug)
        with open(f"debug_page_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(response_text)

        # Parsing QnA hasil ekstraksi
        for block in response_text.split("\n\n"):
            if block.startswith("Q:") and "\nA:" in block:
                try:
                    q = block.split("\nA:")[0].replace("Q: ", "").strip()
                    a = block.split("\nA:")[1].strip()

                    bank, tahun = extract_bank_and_year_from_question(q)
                    all_qna.append([q, a, bank, tahun, "Laporan Tidak Diketahui"])

                    documents.append(Document(
                        page_content=f"Q: {q}\nA: {a}",
                        metadata={
                            "bank": bank,
                            "tahun": tahun,
                            "jenis_laporan": "Laporan Tidak Diketahui",
                            "page": i + 1,
                            "source": os.path.basename(pdf_path)
                        }
                    ))

                except Exception as parse_err:
                    print(f"Gagal parsing QnA di halaman {i + 1}: {parse_err}")
                    continue
                
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pertanyaan", "Jawaban", "Bank", "Tahun", "Jenis Laporan"])
        writer.writerows(all_qna)

    print(f"Semua Q&A disimpan di: {output_path}")
    print(f"Total dokumen yang dihasilkan: {len(documents)}")

    return documents
