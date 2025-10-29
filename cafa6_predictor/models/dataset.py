"""
PyTorch dataset for CAFA-6 training
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple


class CAFA6Dataset(Dataset):
    """Dataset for CAFA-6 protein embeddings and labels"""
    
    def __init__(self, embeddings: np.ndarray, protein_ids: List[str], 
                 labels: Dict[str, np.ndarray] = None):
        """
        Initialize dataset
        
        Args:
            embeddings: Protein embeddings [n_proteins, embedding_dim]
            protein_ids: List of protein IDs
            labels: Optional dict mapping ontology to label matrices
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.protein_ids = protein_ids
        
        if labels is not None:
            self.labels = {
                onto: torch.tensor(labels[onto], dtype=torch.float32)
                for onto in ['MFO', 'BPO', 'CCO']
            }
        else:
            self.labels = None
    
    def __len__(self) -> int:
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.embeddings[idx], self.protein_ids[idx]
        
        return self.embeddings[idx], {
            onto: self.labels[onto][idx]
            for onto in ['MFO', 'BPO', 'CCO']
        }
