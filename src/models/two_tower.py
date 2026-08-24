import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, user_ids):
        u_emb = self.user_emb(user_ids)
        return F.normalize(self.fc(u_emb), p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, item_ids):
        i_emb = self.item_emb(item_ids)
        return F.normalize(self.fc(i_emb), p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)
        
    def forward(self, user_ids, item_ids):
        user_vectors = self.user_tower(user_ids)
        item_vectors = self.item_tower(item_ids)
        return torch.sum(user_vectors * item_vectors, dim=1)

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) Loss function for implicit feedback representation."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
