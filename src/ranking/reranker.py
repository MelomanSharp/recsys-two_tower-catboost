import numpy as np

class MaximalMarginalRelevance:
    """Mitigates general business Popularity Bias using a dynamic diversity filter."""
    @staticmethod
    def balance_diversity(candidates, scores, item_embeddings, top_n=10, lmbda=0.7):
        if len(candidates) == 0:
            return []
            
        selected = [candidates[0]]
        remaining = list(candidates[1:])
        remaining_scores = list(scores[1:])
        
        while len(selected) < top_n and len(remaining) > 0:
            mmr_scores = []
            for i, cand in enumerate(remaining):
                # Calculate maximum similarity with already chosen items
                sim_to_selected = max([np.dot(item_embeddings[cand], item_embeddings[s]) for s in selected])
                score = lmbda * remaining_scores[i] - (1 - lmbda) * sim_to_selected
                mmr_scores.append(score)
                
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
            remaining_scores.pop(best_idx)
            
        return selected
