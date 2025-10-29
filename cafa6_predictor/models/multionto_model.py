"""
Multi-ontology model with separate heads for MFO/BPO/CCO
"""
import torch
import torch.nn as nn
from typing import Dict


class Head(nn.Module):
    """Single ontology prediction head"""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiOntoModel(nn.Module):
    """Multi-ontology model with separate heads for MFO, BPO, CCO"""
    
    def __init__(self, embedding_dim: int, ontology_sizes: Dict[str, int], 
                 hidden_dim: int = 1024, dropout: float = 0.2):
        """
        Initialize multi-ontology model
        
        Args:
            embedding_dim: Input embedding dimension
            ontology_sizes: Dict mapping ontology to number of terms
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.mfo = Head(embedding_dim, ontology_sizes['MFO'], hidden_dim, dropout)
        self.bpo = Head(embedding_dim, ontology_sizes['BPO'], hidden_dim, dropout)
        self.cco = Head(embedding_dim, ontology_sizes['CCO'], hidden_dim, dropout)
        
        self.ontology_sizes = ontology_sizes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input embeddings [batch, embedding_dim]
            
        Returns:
            Dict mapping ontology to logits [batch, num_terms]
        """
        return {
            'MFO': self.mfo(x),
            'BPO': self.bpo(x),
            'CCO': self.cco(x)
        }


def build_model(go_loader, config) -> MultiOntoModel:
    """
    Build model from configuration
    
    Args:
        go_loader: GOGraphLoader instance
        config: Model configuration
        
    Returns:
        MultiOntoModel instance
    """
    ontology_sizes = {
        onto: len(go_loader.ontology_terms[onto]) 
        for onto in ['MFO', 'BPO', 'CCO']
    }
    
    model = MultiOntoModel(
        embedding_dim=config.embedding_dim,
        ontology_sizes=ontology_sizes,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )
    
    return model
