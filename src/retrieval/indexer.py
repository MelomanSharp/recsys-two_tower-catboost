import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.models.two_tower import TwoTowerModel, BPRLoss
from src.retrieval.faiss_cache import FaissIndexer
from src.config import Config
from tqdm import tqdm 

class InteractionDataset(Dataset):
    """Simple dataset for (user, positive item, negative item) triples."""
    def __init__(self, user_ids, item_ids, neg_item_ids):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.neg_item_ids = neg_item_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.neg_item_ids[idx]

class TwoTowerIndexer:
    def __init__(self, num_users, num_items, user_encoder, item_encoder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoTowerModel(num_users, num_items, Config.EMBEDDING_DIM).to(self.device)
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.faiss_index = FaissIndexer(Config.EMBEDDING_DIM)
        
    def train(self, transactions, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE):
        user_ids = transactions['customer_id_enc'].values
        pos_item_ids = transactions['article_id_enc'].values
        
        # Negative sampling: randomly pick items from the entire catalogue
        all_items = np.arange(self.model.item_tower.item_emb.num_embeddings)
        neg_item_ids = np.random.choice(all_items, size=len(user_ids))
        
        dataset = InteractionDataset(user_ids, pos_item_ids, neg_item_ids)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = BPRLoss()
        
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            epoch_loss = 0
            for u_batch, i_pos_batch, i_neg_batch in loader:
                u_batch = u_batch.to(self.device)
                i_pos_batch = i_pos_batch.to(self.device)
                i_neg_batch = i_neg_batch.to(self.device)
                
                optimizer.zero_grad()
                pos_scores = self.model(u_batch, i_pos_batch)
                neg_scores = self.model(u_batch, i_neg_batch)
                
                loss = criterion(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, BPR Loss: {epoch_loss/len(loader):.4f}")

    def build_item_index(self):
        """Extract embeddings for all items and build FAISS index."""
        self.model.eval()
        with torch.no_grad():
            num_items = self.model.item_tower.item_emb.num_embeddings
            all_item_ids = torch.arange(num_items).to(self.device)
            
            chunk_size = 1024
            all_embs = []
            for i in range(0, num_items, chunk_size):
                chunk = all_item_ids[i:i+chunk_size]
                embs = self.model.item_tower(chunk).cpu().numpy()
                all_embs.append(embs)
            
            item_embeddings = np.vstack(all_embs)
            
        # Exclude padding index (0) – we reserved that for unknown items
        original_item_ids = self.item_encoder.classes_
        valid_embeddings = item_embeddings[1:] 
        
        self.faiss_index.add_items(original_item_ids, valid_embeddings)
        return valid_embeddings

    def get_user_embedding(self, user_id_encoded):
        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([user_id_encoded], dtype=torch.long).to(self.device)
            emb = self.model.user_tower(u_tensor).cpu().numpy()
        return emb