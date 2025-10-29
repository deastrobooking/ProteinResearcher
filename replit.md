# CAFA-6 Protein Function Prediction System

## Project Overview
This is a modular, CAFA-6 compliant implementation for protein function prediction. The system implements a baseline hybrid predictor combining protein language model embeddings with multi-ontology classification, IC-weighted evaluation, and ancestor closure.

## Architecture
The project follows a 5-tier modular architecture:
1. **Data Ingestion** (`data_ingest/`) - GO graph loading, label building, IA weights
2. **Models** (`models/`) - Multi-ontology neural network with separate MFO/BPO/CCO heads
3. **Evaluation** (`evaluation/`) - IC-weighted maxF1 metrics and ancestor closure
4. **Configuration** (`config/`) - Centralized configuration management
5. **Orchestration** (`main.py`) - End-to-end training pipeline

## Key Design Decisions
- **Ontology-specific heads**: Separate classifiers for MFO/BPO/CCO to handle different term distributions
- **BCEWithLogitsLoss**: Proper multilabel loss with positive class weighting for imbalance
- **Ancestor closure**: Applied after scoring, before thresholding (critical for CAFA metrics)
- **Caching**: Aggressive caching of GO graph, labels, and embeddings to speed up iterations
- **CAFA-6 compliance**: Strict adherence to submission format (≤1500 terms, ≤3 sig figs, tab-separated)

## Current State
✅ Baseline implementation complete (Phase 1)
- GO ontology loading and graph management
- Label matrix construction from train_terms.tsv
- IA weight loading and per-ontology indexing
- Multi-head model with BCEWithLogitsLoss
- IC-weighted maxF1 evaluation with ancestor closure
- CAFA-6 compliant submission writer
- Demo mode with synthetic data

## Recent Changes (2025-10-29)
- Created complete project structure
- Implemented all Phase 1 baseline components
- Added demo mode for testing without real data
- Configured caching system for expensive operations
- Set up training pipeline with validation split
- Implemented IC-weighted threshold optimization

## Next Steps (Phase 2+)
- DIAMOND homology integration for sequence similarity features
- Zero-shot GO term encoding for rare labels
- Multi-pLM ensemble (ESM-2 + ProtT5)
- Hierarchical loss regularization
- Species-aware cross-validation
- Production deployment configuration

## How to Use
**Demo mode** (no data required):
```bash
python cafa6_predictor/main.py --demo
```

**With real CAFA-6 data**:
Place data files in `cafa6_predictor/data/` and run:
```bash
python cafa6_predictor/main.py
```

## Dependencies
Python 3.11+ with torch, torchmetrics, obonet, networkx, numpy, pandas, biopython, scikit-learn, tqdm
