"""
Information Accretion (IA) weight loader
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pickle


class IAWeightLoader:
    """Load and manage IA weights for IC-weighted metrics"""
    
    def __init__(self, go_loader, cache_dir: Path = None):
        """
        Initialize IA weight loader
        
        Args:
            go_loader: GOGraphLoader instance
            cache_dir: Optional cache directory
        """
        self.go_loader = go_loader
        self.cache_dir = cache_dir
        self.ia_dict = {}
        self.ia_vectors = {}
        
    def load(self, ia_path: Path, use_cache: bool = True) -> 'IAWeightLoader':
        """
        Load IA weights from TSV file
        
        Args:
            ia_path: Path to IA.tsv
            use_cache: Whether to use/create cache
            
        Returns:
            Self for chaining
        """
        cache_path = self.cache_dir / "ia_weights.pkl" if self.cache_dir else None
        
        if use_cache and cache_path and cache_path.exists():
            print(f"Loading IA weights from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.ia_dict = cached['ia_dict']
                self.ia_vectors = cached['ia_vectors']
            print(f"✓ Loaded {len(self.ia_dict)} IA weights from cache")
            return self
        
        print(f"Loading IA weights from: {ia_path}")
        
        df = pd.read_csv(ia_path, sep='\t', header=None, names=['term', 'ia'])
        self.ia_dict = dict(zip(df['term'], df['ia']))
        
        for onto in ['MFO', 'BPO', 'CCO']:
            terms = self.go_loader.ontology_terms[onto]
            ia_vec = np.array([self.ia_dict.get(t, 0.0) for t in terms], dtype=np.float32)
            self.ia_vectors[onto] = ia_vec
            print(f"  - {onto}: mean IA = {ia_vec.mean():.4f}, range = [{ia_vec.min():.4f}, {ia_vec.max():.4f}]")
        
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'ia_dict': self.ia_dict,
                    'ia_vectors': self.ia_vectors
                }, f)
            print(f"✓ Cached IA weights to {cache_path}")
        
        print(f"✓ Loaded {len(self.ia_dict)} IA weights")
        return self
