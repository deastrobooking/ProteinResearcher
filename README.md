# CAFA-6 Protein Function Prediction System

A modular, CAFA-6 compliant implementation for protein function prediction using protein language models and GO ontology.

## Architecture Overview

This system implements a hybrid prediction approach combining:
- **Protein Language Model Embeddings** (ESM-2/ProtT5)
- **Multi-ontology Classification** (separate heads for MFO/BPO/CCO)
- **IC-weighted Evaluation** (Information Accretion weighted maxF1)
- **Ancestor Closure** (GO graph propagation)
- **CAFA-6 Compliant Submission** (≤1500 terms, ≤3 sig figs, tab-separated)

## Project Structure

```
cafa6_predictor/
├── config/              # Configuration management
│   └── base.py         # PathConfig, ModelConfig, EvaluationConfig
├── data_ingest/        # Data loading and preprocessing
│   ├── go_loader.py    # GO ontology graph management
│   ├── label_builder.py # Multi-hot label matrix construction
│   └── ia_weights.py   # Information Accretion weight loading
├── models/             # Neural network architectures
│   ├── multionto_model.py # 3-head model (MFO/BPO/CCO)
│   ├── dataset.py      # PyTorch dataset
│   └── trainer.py      # Training loop
├── evaluation/         # Metrics and submission
│   ├── metrics.py      # IC-weighted maxF1 with ancestor closure
│   └── submission.py   # CAFA-6 compliant file writer
├── data/               # Data files (not in repo)
└── main.py             # Main training pipeline
```

## Key Features

### ✅ CAFA-6 Compliance
- Uses correct GO release (2025-06-01)
- Builds labels from train_terms.tsv (not pre-computed matrices)
- BCEWithLogitsLoss for multilabel classification
- Proper ancestor closure before thresholding
- IC-weighted metrics matching official CAFA scoring
- Submission format validation

### ✅ Modular Design
- Separate components for data, models, evaluation
- Caching for expensive operations (GO graph, labels, embeddings)
- Configurable via dataclass configs
- Easy to extend with new features (homology, zero-shot)

### ✅ Robust Training
- Per-ontology positive class weighting
- Gradient clipping
- Validation-based threshold tuning
- Checkpoint saving

## Quick Start

### Demo Mode (No Data Required)
```bash
python cafa6_predictor/main.py --demo
```

This will:
1. Create minimal dummy GO ontology and labels
2. Generate random embeddings
3. Train a small model
4. Evaluate with IC-weighted metrics

### With Real Data
Place CAFA-6 data files in `cafa6_predictor/data/`:
- `go-basic.obo`
- `train_terms.tsv`
- `IA.tsv`
- `train_embeddings.npy` (pre-computed ESM-2 embeddings)
- `train_ids.npy`
- `test_embeddings.npy`
- `test_ids.npy`

Then run:
```bash
python cafa6_predictor/main.py
```

## Configuration

Edit `cafa6_predictor/config/base.py` to customize:
- **Paths**: Data directories
- **Model**: Architecture (embedding_dim, hidden_dim, dropout)
- **Training**: Learning rate, batch size, epochs
- **Evaluation**: Threshold search, validation split

## Key Components

### 1. GO Graph Loader
```python
from data_ingest import GOGraphLoader

go_loader = GOGraphLoader(obo_path, cache_dir)
go_loader.load()
# Access: go_loader.ontology_terms, go_loader.ancestor_indices
```

### 2. Label Builder
```python
from data_ingest import LabelBuilder

label_builder = LabelBuilder(go_loader, cache_dir)
label_builder.build_from_tsv(train_terms_path)
# Access: label_builder.labels['MFO'], label_builder.protein_ids
```

### 3. Multi-Ontology Model
```python
from models import build_model

model = build_model(go_loader, config.model)
# Outputs: {'MFO': logits, 'BPO': logits, 'CCO': logits}
```

### 4. IC-Weighted Evaluation
```python
from evaluation import AncestorClosure, ICWeightedMaxF1

closure = AncestorClosure(ancestor_indices)
metric = ICWeightedMaxF1(ia_vector, closure)
best_f1, threshold = metric.find_best_threshold(scores, labels, thresholds)
```

### 5. Submission Writer
```python
from evaluation import SubmissionWriter

SubmissionWriter.write_submission(
    protein_ids, predictions, go_terms, 
    thresholds, ancestor_closures, output_path
)
```

## Next Steps (Phase 2+)

- [ ] DIAMOND homology integration
- [ ] Zero-shot GO term encoding
- [ ] Multi-pLM ensemble (ESM-2 + ProtT5)
- [ ] Hierarchical loss regularization
- [ ] Species-aware cross-validation
- [ ] Calibration (Platt/isotonic scaling)

## References

Based on CAFA-6 competition best practices:
- Hybrid predictor (homology + pLM + GO-aware)
- IC-weighted maxF1 optimization
- Ancestor closure enforcement
- Compliant submission formatting

## Requirements

- Python 3.11+
- PyTorch
- torchmetrics
- obonet
- networkx
- numpy
- pandas
- biopython
- scikit-learn
- tqdm
