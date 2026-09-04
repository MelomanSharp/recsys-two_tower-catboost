from src.config import Config

class CandidateSearcher:
    def __init__(self, faiss_index, user_encoder, item_encoder):
        self.faiss_index = faiss_index
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def get_candidates(self, user_embedding, top_k=Config.TOP_K_CANDIDATES):
        """Return candidate item IDs for a user."""
        candidates = self.faiss_index.search_candidates(user_embedding, top_k=top_k)
        return candidates[0] if candidates else []