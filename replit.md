# CAFA-6 Protein Function Prediction System

## Project Overview
This is a modular, CAFA-6 compliant implementation for protein function prediction. The system implements a baseline hybrid predictor combining protein language model embeddings with multi-ontology classification, IC-weighted evaluation, and ancestor closure.

## Architecture
The project follows a 7-tier modular architecture:
1. **Data Ingestion** (`data_ingest/`) - GO graph loading, label building, IA weights
2. **Homology** (`homology/`) - DIAMOND sequence alignment and homology features
3. **Zero-Shot** (`zero_shot/`) - GO term encoding with sentence transformers
4. **Models** (`models/`) - Hybrid multi-ontology architecture with linear + zero-shot heads
5. **Evaluation** (`evaluation/`) - IC-weighted maxF1 metrics and ancestor closure
6. **Configuration** (`config/`) - Centralized configuration management
7. **Orchestration** (`main.py`) - End-to-end training pipeline

## Key Design Decisions
- **Ontology-specific heads**: Separate classifiers for MFO/BPO/CCO to handle different term distributions
- **BCEWithLogitsLoss**: Proper multilabel loss with positive class weighting for imbalance
- **Ancestor closure**: Applied after scoring, before thresholding (critical for CAFA metrics)
- **Homology integration**: DIAMOND BLASTP + feature concatenation (1280 pLM + 128 homology = 1408-dim)
- **Zero-shot learning**: Bilinear matching between protein embeddings and GO term text embeddings (384-dim)
- **Hybrid architecture**: Combines linear heads + zero-shot heads with alpha=0.5 weighting for best of both worlds
- **Caching**: Aggressive caching of GO graph, labels, embeddings, DIAMOND database, alignments, and term embeddings
- **CAFA-6 compliance**: Strict adherence to submission format (≤1500 terms, ≤3 sig figs, tab-separated)

## Current State
✅ **Phase 1 Complete** - CAFA-6 Baseline
- GO ontology loading and graph management
- Label matrix construction from train_terms.tsv
- IA weight loading and per-ontology indexing
- Multi-head model with BCEWithLogitsLoss
- IC-weighted maxF1 evaluation with ancestor closure
- CAFA-6 compliant submission writer
- Demo mode with synthetic data

✅ **Phase 2 Complete** - DIAMOND Homology Integration
- DIAMOND v2.1.11 sequence aligner installed
- DiamondRunner: database building and BLASTP execution
- HomologyFeatureExtractor: GO term transfer and feature extraction
- 128-dim homology features combined with 1280-dim pLM embeddings
- Model dynamically adapts to 1408-dim input
- CLI flags: --no-homology, --rebuild-diamond
- Full caching support for DIAMOND database and alignments

✅ **Phase 3 Complete** - Zero-Shot GO Term Encoding
- Sentence transformer integration (all-MiniLM-L6-v2) for GO term text embeddings
- GOTermEncoder: extracts GO descriptions from go-basic.obo and generates 384-dim embeddings
- ZeroShotHead: bilinear matching layer for protein-term similarity scoring
- HybridMultiOntoModel: combines linear heads + zero-shot heads with learnable weighting
- Three prediction modes: pure linear, pure zero-shot, hybrid (alpha=0.5)
- Term embedding caching for fast reloading
- CLI flags: --use-zero-shot, --use-hybrid
- Model scales to 5.5M parameters in hybrid mode

## Recent Changes (2025-10-29)
**Phase 1 (Baseline):**
- Created complete project structure with 5 core modules
- Implemented all baseline components (GO loader, labels, model, evaluation)
- Added demo mode for testing without real data
- Configured caching system for expensive operations
- Set up training pipeline with validation split

**Phase 2 (DIAMOND Homology):**
- Installed DIAMOND v2.1.11 for fast sequence alignment
- Implemented DiamondRunner for database building and BLASTP
- Implemented HomologyFeatureExtractor for GO transfer and feature extraction
- Integrated homology features into main pipeline (1280 → 1408-dim embeddings)
- Added configuration options and CLI flags for homology control
- Extended caching to DIAMOND database and alignment results

**Phase 3 (Zero-Shot GO Term Encoding):**
- Installed transformers, sentencepiece, and sentence-transformers
- Implemented GOTermEncoder for text-based GO term embeddings (384-dim)
- Implemented ZeroShotHead with bilinear matching (also supports MLP, cosine)
- Created HybridMultiOntoModel combining linear + zero-shot heads
- Added ZeroShotConfig to centralized configuration
- Integrated zero-shot into main pipeline with conditional encoding
- Tested hybrid mode successfully: 5.5M parameters, IC-weighted maxF1 = 1.0000
- Fixed huggingface-hub version conflict (<1.0 required)

## Next Steps (Phase 4+)
- Multi-pLM ensemble (ESM-2 + ProtT5 + Ankh)
- Hierarchical loss regularization (penalize ontology violations)
- Species-aware cross-validation
- Advanced homology transfer (weighted voting, propagation)
- Calibration and uncertainty quantification
- Production deployment configuration
- Validation on realistic CAFA dataset split
- Regression tests for caching behavior

## How to Use
**Demo mode** (no data required):
```bash
python cafa6_predictor/main.py --demo
```

**With homology disabled** (pLM only):
```bash
python cafa6_predictor/main.py --demo --no-homology
```

**Force rebuild DIAMOND database**:
```bash
python cafa6_predictor/main.py --demo --rebuild-diamond
```

**Hybrid mode** (linear + zero-shot):
```bash
python cafa6_predictor/main.py --demo --use-hybrid
```

**Pure zero-shot mode** (no linear heads):
```bash
python cafa6_predictor/main.py --demo --use-zero-shot
```

**With real CAFA-6 data**:
Place data files in `cafa6_predictor/data/` and run:
```bash
python cafa6_predictor/main.py
```

Required data files:
- `go-basic.obo` - GO ontology
- `train_terms.tsv` - Protein-GO term annotations
- `train_sequences.fasta` - Protein sequences (for DIAMOND)
- `IA.tsv` - Information Accretion weights
- `train_embeddings.npy` - ESM-2 embeddings (1280-dim)
- `train_ids.npy` - Protein IDs
- `test_embeddings.npy` - Test embeddings
- `test_ids.npy` - Test IDs

## Dependencies
**Python packages:**
- torch, torchmetrics - Deep learning
- obonet, networkx - GO graph processing
- numpy, pandas - Data manipulation
- biopython - Sequence I/O
- scikit-learn - Utilities
- tqdm - Progress bars
- transformers, sentencepiece - Zero-shot text encoding
- huggingface-hub (<1.0) - Model downloading

**System packages:**
- DIAMOND v2.1.11 - Sequence alignment
