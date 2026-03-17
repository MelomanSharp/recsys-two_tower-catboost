import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, embedding_dim):
        # Cosine alignment similarity via inner-product structure
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.item_ids = []
        
    def add_items(self, item_ids, embeddings):
        """Registers new item vectors within the active indexing structure."""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))
        self.item_ids.extend(item_ids)
        
    def search_candidates(self, user_embeddings, top_k=100):
        """Fetches top structural nearest neighbors based on vector similarity alignment."""
        faiss.normalize_L2(user_embeddings)
        distances, indices = self.index.search(user_embeddings.astype("float32"), top_k)
        
        results = []
        for row in indices:
            results.append([self.item_ids[idx] for idx in row if idx != -1])
        return results
