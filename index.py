import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class DenseIndexer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
    
    def encode(self, texts, batch_size=16):
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0]  # CLS
            vectors.append(emb.cpu().numpy())
        return np.vstack(vectors)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index