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
    """Model architecture and training configuration"""
    embedding_dim: int = 1280
    hidden_dim: int = 512
    dropout_rate: float = 0.3

    batch_size: int = 32
    num_epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    device: str = 'cpu'

    def optimize_batch_size(self, device: str):
        """Increase batch size for GPU training"""
        if device == 'cuda':
            self.batch_size = 128  # Larger batches for GPU
        return self.batch_size


@dataclass
class HomologyConfig:
    """Homology search configuration"""
    use_homology: bool = True
    homology_feature_dim: int = 128

    max_target_seqs: int = 100
    evalue: float = 1e-3
    sensitivity: str = 'sensitive'
    threads: int = 4

    min_identity: float = 30.0
    min_coverage: float = 0.5
    top_k_hits: int = 10


@dataclass
class ZeroShotConfig:
    """Zero-shot GO term encoding configuration"""
    use_zero_shot: bool = False
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    method: str = 'bilinear'

    use_hybrid: bool = False
    hybrid_alpha: float = 0.5


@dataclass
class EnsembleConfig:
    """Multi-pLM ensemble configuration"""
    use_ensemble: bool = False
    fusion_strategy: str = 'concat'

    plm_models: Dict = field(default_factory=lambda: {
        'esm2': {'file': 'train_embeddings.npy', 'dim': 1280, 'enabled': True},
        'prott5': {'file': 'train_embeddings_prott5.npy', 'dim': 1024, 'enabled': False},
        'ankh': {'file': 'train_embeddings_ankh.npy', 'dim': 768, 'enabled': False},
    })

    output_projection_dim: int = None


@dataclass
class OptimizationConfig:
    """Phase 5: Advanced optimization configuration"""
    use_topk_filter: bool = True
    use_hierarchical_loss: bool = False
    hierarchical_loss_weight: float = 0.1

    threshold_strategy: str = 'per_ontology'
    fine_grained_search: bool = False
    threshold_search_points: int = 99


@dataclass
class EvaluationConfig:
    """Evaluation and submission configuration"""
    max_terms_per_protein: int = 1500
    threshold_min: float = 0.01
    threshold_max: float = 0.99

    validation_split: float = 0.1
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration object"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    homology: HomologyConfig = field(default_factory=HomologyConfig)
    zero_shot: ZeroShotConfig = field(default_factory=ZeroShotConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    ontologies: tuple = ('MFO', 'BPO', 'CCO')

    def __post_init__(self):
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.ensemble.use_ensemble:
            enabled_models = {k: v for k, v in self.ensemble.plm_models.items() if v['enabled']}
            if not enabled_models:
                raise ValueError("No pLM models enabled in ensemble configuration")

            if self.ensemble.fusion_strategy == 'concat':
                base_dim = sum(v['dim'] for v in enabled_models.values())
            else:
                base_dim = list(enabled_models.values())[0]['dim']

            if self.ensemble.output_projection_dim:
                base_dim = self.ensemble.output_projection_dim

            if self.homology.use_homology:
                self.model.embedding_dim = base_dim + self.homology.homology_feature_dim
            else:
                self.model.embedding_dim = base_dim
        elif self.homology.use_homology:
            self.model.embedding_dim = 1280 + self.homology.homology_feature_dim