from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

EMBED_MODEL = 'all-MiniLM-L6-v2'

def embed_chunks(chunks, model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
