"""
Base configuration for CAFA-6 prediction pipeline
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PathConfig:
    """Data paths configuration"""
    base_dir: Path = Path("cafa6_predictor/data")
    go_obo: Path = field(default_factory=lambda: Path("cafa6_predictor/data/go-basic.obo"))
    train_terms: Path = field(default_factory=lambda: Path("cafa6_predictor/data/train_terms.tsv"))
    train_sequences: Path = field(default_factory=lambda: Path("cafa6_predictor/data/train_sequences.fasta"))
    test_sequences: Path = field(default_factory=lambda: Path("cafa6_predictor/data/testsuperset.fasta"))
    ia_weights: Path = field(default_factory=lambda: Path("cafa6_predictor/data/IA.tsv"))
    train_taxonomy: Path = field(default_factory=lambda: Path("cafa6_predictor/data/train_taxonomy.tsv"))
    
    train_embeddings: Path = field(default_factory=lambda: Path("cafa6_predictor/data/train_embeddings.npy"))
    train_ids: Path = field(default_factory=lambda: Path("cafa6_predictor/data/train_ids.npy"))
    test_embeddings: Path = field(default_factory=lambda: Path("cafa6_predictor/data/test_embeddings.npy"))
    test_ids: Path = field(default_factory=lambda: Path("cafa6_predictor/data/test_ids.npy"))
    
    cache_dir: Path = field(default_factory=lambda: Path("cafa6_predictor/cache"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("cafa6_predictor/checkpoints"))


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    embedding_dim: int = 1280
    hidden_dim: int = 1024
    dropout: float = 0.2
    
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 6
    gradient_clip: float = 1.0
    
    device: str = "cpu"


@dataclass
class EvaluationConfig:
    """Evaluation and submission configuration"""
    max_terms_per_protein: int = 1500
    threshold_search_points: int = 99
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    
    validation_split: float = 0.1
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration object"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    ontologies: tuple = ('MFO', 'BPO', 'CCO')
    
    def __post_init__(self):
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
