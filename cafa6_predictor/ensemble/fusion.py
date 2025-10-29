"""
Multi-pLM Ensemble: Fuse embeddings from multiple protein language models.

Supports ESM-2, ProtT5, Ankh, and other pLMs for improved protein representations.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle


class EmbeddingFusion(nn.Module):
    """
    Fuse embeddings from multiple protein language models.
    
    Strategies:
    - concat: Simple concatenation [ESM-2 | ProtT5 | Ankh]
    - weighted_avg: Learnable weighted average
    - attention: Attention-based fusion with learnable query
    - gated: Gated fusion with learned gate weights
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        fusion_strategy: str = 'concat',
        output_dim: Optional[int] = None
    ):
        """
        Args:
            embedding_dims: Dict mapping pLM name -> embedding dimension
                           e.g., {'esm2': 1280, 'prott5': 1024, 'ankh': 768}
            fusion_strategy: 'concat', 'weighted_avg', 'attention', or 'gated'
            output_dim: Optional projection dimension (None = keep native dim)
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.fusion_strategy = fusion_strategy
        self.plm_names = list(embedding_dims.keys())
        self.n_models = len(self.plm_names)
        
        # Calculate total dimension based on strategy
        if fusion_strategy == 'concat':
            self.fused_dim = sum(embedding_dims.values())
        else:
            # weighted_avg, attention, gated all output same dim as first model
            self.fused_dim = list(embedding_dims.values())[0]
        
        # Build fusion layers
        if fusion_strategy == 'weighted_avg':
            # Learnable weights (softmax normalized)
            self.weights = nn.Parameter(torch.ones(self.n_models))
        
        elif fusion_strategy == 'attention':
            # Attention-based fusion
            self.query = nn.Parameter(torch.randn(self.fused_dim))
            # Project each embedding to common dimension if needed
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, self.fused_dim) if dim != self.fused_dim else nn.Identity()
                for name, dim in embedding_dims.items()
            })
        
        elif fusion_strategy == 'gated':
            # Gated fusion with learned gates
            self.gates = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.Sigmoid()
                )
                for name, dim in embedding_dims.items()
            })
            # Project to common dimension
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, self.fused_dim) if dim != self.fused_dim else nn.Identity()
                for name, dim in embedding_dims.items()
            })
        
        # Optional output projection
        self.output_projection = None
        if output_dim is not None:
            self.output_projection = nn.Linear(self.fused_dim, output_dim)
            self.fused_dim = output_dim
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse embeddings from multiple pLMs.
        
        Args:
            embeddings: Dict mapping pLM name -> tensor of shape (batch, dim)
        
        Returns:
            Fused embedding tensor of shape (batch, fused_dim)
        """
        # Ensure all expected models are present
        for name in self.plm_names:
            if name not in embeddings:
                raise ValueError(f"Missing embedding for model: {name}")
        
        if self.fusion_strategy == 'concat':
            # Simple concatenation
            emb_list = [embeddings[name] for name in self.plm_names]
            fused = torch.cat(emb_list, dim=-1)
        
        elif self.fusion_strategy == 'weighted_avg':
            # Softmax-normalized weighted average
            weights = torch.softmax(self.weights, dim=0)
            emb_list = [embeddings[name] for name in self.plm_names]
            fused = sum(w * emb for w, emb in zip(weights, emb_list))
        
        elif self.fusion_strategy == 'attention':
            # Attention-based fusion
            # Project all embeddings to common dimension
            projected = [self.projections[name](embeddings[name]) for name in self.plm_names]
            # Stack: (batch, n_models, dim)
            stacked = torch.stack(projected, dim=1)
            # Attention scores: dot product with query
            scores = torch.einsum('bnd,d->bn', stacked, self.query)
            attn_weights = torch.softmax(scores, dim=1)
            # Weighted sum
            fused = torch.einsum('bn,bnd->bd', attn_weights, stacked)
        
        elif self.fusion_strategy == 'gated':
            # Gated fusion
            gated_embs = []
            for name in self.plm_names:
                emb = embeddings[name]
                gate = self.gates[name](emb)
                gated = gate * emb
                projected = self.projections[name](gated)
                gated_embs.append(projected)
            # Average gated embeddings
            fused = torch.stack(gated_embs, dim=0).mean(dim=0)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Apply output projection if configured
        if self.output_projection is not None:
            fused = self.output_projection(fused)
        
        return fused
    
    def get_output_dim(self) -> int:
        """Get the output dimension after fusion."""
        return self.fused_dim


def load_multi_embeddings(
    data_dir: Path,
    protein_ids: np.ndarray,
    plm_configs: Dict[str, Dict],
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load embeddings from multiple protein language models.
    
    Args:
        data_dir: Directory containing embedding files
        protein_ids: Array of protein IDs to load
        plm_configs: Dict mapping pLM name -> config dict with keys:
                    - 'file': embedding filename (e.g., 'train_embeddings_esm2.npy')
                    - 'dim': embedding dimension
        cache_dir: Optional cache directory for processed embeddings
        use_cache: Whether to use cached embeddings
    
    Returns:
        Tuple of (fused_embeddings, embedding_dims)
        - fused_embeddings: numpy array of shape (n_proteins, total_dim)
        - embedding_dims: Dict mapping pLM name -> dimension
    """
    embeddings_dict = {}
    embedding_dims = {}
    
    for plm_name, config in plm_configs.items():
        emb_file = data_dir / config['file']
        
        if not emb_file.exists():
            print(f"⚠ {plm_name} embeddings not found: {emb_file}")
            print(f"  Skipping {plm_name} in ensemble")
            continue
        
        # Load embeddings
        embeddings = np.load(emb_file)
        
        # Verify dimension
        if embeddings.shape[1] != config['dim']:
            print(f"⚠ Warning: {plm_name} dimension mismatch")
            print(f"  Expected: {config['dim']}, Got: {embeddings.shape[1]}")
        
        embeddings_dict[plm_name] = embeddings
        embedding_dims[plm_name] = embeddings.shape[1]
        
        print(f"✓ Loaded {plm_name}: {embeddings.shape}")
    
    if not embeddings_dict:
        raise ValueError("No valid embeddings found for any pLM")
    
    return embeddings_dict, embedding_dims


def create_demo_multi_embeddings(
    n_proteins: int,
    plm_dims: Dict[str, int],
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create synthetic multi-pLM embeddings for demo mode.
    
    Args:
        n_proteins: Number of proteins
        plm_dims: Dict mapping pLM name -> embedding dimension
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping pLM name -> embeddings array (n_proteins, dim)
    """
    np.random.seed(seed)
    embeddings = {}
    
    for plm_name, dim in plm_dims.items():
        # Generate random embeddings with different distributions
        # to simulate different pLM characteristics
        if 'esm' in plm_name.lower():
            # ESM-2: high-dimensional, normalized
            emb = np.random.randn(n_proteins, dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        elif 'prott5' in plm_name.lower():
            # ProtT5: medium-dimensional, centered
            emb = np.random.randn(n_proteins, dim).astype(np.float32) * 0.5
        elif 'ankh' in plm_name.lower():
            # Ankh: lower-dimensional, sparse
            emb = np.random.randn(n_proteins, dim).astype(np.float32)
            emb = np.where(np.abs(emb) < 0.5, 0, emb)
        else:
            # Default: standard normal
            emb = np.random.randn(n_proteins, dim).astype(np.float32)
        
        embeddings[plm_name] = emb
    
    return embeddings
