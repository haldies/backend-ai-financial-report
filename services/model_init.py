import torch
import os
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from google import genai
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.voyageai import VoyageEmbedding
load_dotenv()

VOYAGE_API = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = "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"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default-collection")


llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
embed_model = VoyageEmbedding(
    voyage_api_key=VOYAGE_API,
    model_name="voyage-3-large",
)

# embed_model = HuggingFaceEmbedding(
#     model_name=EMBED_MODEL,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=1000.0 
)


vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME
)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

print("✅ Model initialized")
