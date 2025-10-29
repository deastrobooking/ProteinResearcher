"""
Build multi-hot label matrices from train_terms.tsv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


class LabelBuilder:
    """Build and manage training labels"""
    
    def __init__(self, go_loader, cache_dir: Path = None):
        """
        Initialize label builder
        
        Args:
            go_loader: GOGraphLoader instance
            cache_dir: Optional cache directory
        """
        self.go_loader = go_loader
        self.cache_dir = cache_dir
        self.protein_ids = []
        self.protein_index = {}
        self.labels = {}
        
    def build_from_tsv(self, train_terms_path: Path, use_cache: bool = True) -> 'LabelBuilder':
        """
        Build label matrices from train_terms.tsv
        
        Args:
            train_terms_path: Path to train_terms.tsv
            use_cache: Whether to use/create cache
            
        Returns:
            Self for chaining
        """
        cache_path = self.cache_dir / "labels.pkl" if self.cache_dir else None
        
        if use_cache and cache_path and cache_path.exists():
            print(f"Loading labels from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.protein_ids = cached['protein_ids']
                self.protein_index = cached['protein_index']
                self.labels = cached['labels']
            print(f"✓ Loaded labels for {len(self.protein_ids)} proteins")
            return self
        
        print(f"Building labels from: {train_terms_path}")
        
        df = pd.read_csv(train_terms_path, sep='\t', header=None, 
                        names=['protein_id', 'go_term', 'ontology'])
        
        valid_terms = set(self.go_loader.graph.nodes())
        df = df[df['go_term'].isin(valid_terms)]
        
        self.protein_ids = sorted(df['protein_id'].unique())
        self.protein_index = {p: i for i, p in enumerate(self.protein_ids)}
        
        n_proteins = len(self.protein_ids)
        
        for onto in ['MFO', 'BPO', 'CCO']:
            n_terms = len(self.go_loader.ontology_terms[onto])
            matrix = np.zeros((n_proteins, n_terms), dtype=np.uint8)
            
            onto_df = df[df['go_term'].map(self.go_loader.get_ontology) == onto]
            term_idx = self.go_loader.term_indices[onto]
            
            for protein_id, go_term in zip(onto_df['protein_id'], onto_df['go_term']):
                if go_term in term_idx:
                    p_idx = self.protein_index[protein_id]
                    t_idx = term_idx[go_term]
                    matrix[p_idx, t_idx] = 1
            
            self.labels[onto] = matrix
            pos_count = matrix.sum()
            print(f"  - {onto}: {n_terms} terms, {pos_count:,} positive labels")
        
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'protein_ids': self.protein_ids,
                    'protein_index': self.protein_index,
                    'labels': self.labels
                }, f)
            print(f"✓ Cached labels to {cache_path}")
        
        print(f"✓ Built labels for {n_proteins} proteins")
        return self
    
    def compute_pos_weights(self) -> Dict[str, np.ndarray]:
        """
        Compute positive class weights for handling class imbalance
        
        Returns:
            Dict mapping ontology to weight array
        """
        pos_weights = {}
        for onto in ['MFO', 'BPO', 'CCO']:
            matrix = self.labels[onto]
            n_samples = matrix.shape[0]
            pos = matrix.sum(axis=0)
            neg = n_samples - pos
            
            weights = neg / (pos + 1e-6)
            weights[np.isinf(weights)] = 0.0
            weights = np.clip(weights, 0, 100)
            
            pos_weights[onto] = weights.astype(np.float32)
        
        return pos_weights
