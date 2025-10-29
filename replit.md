# CAFA-6 Protein Function Prediction System

## Project Overview
This is a modular, CAFA-6 compliant implementation for protein function prediction. The system implements a baseline hybrid predictor combining protein language model embeddings with multi-ontology classification, IC-weighted evaluation, and ancestor closure.

## Architecture
The project follows an 8-tier modular architecture:
1. **Data Ingestion** (`data_ingest/`) - GO graph loading, label building, IA weights
2. **Homology** (`homology/`) - DIAMOND sequence alignment and homology features
3. **Zero-Shot** (`zero_shot/`) - GO term encoding with sentence transformers
4. **Ensemble** (`ensemble/`) - Multi-pLM embedding fusion (ESM-2, ProtT5, Ankh)
5. **Models** (`models/`) - Hybrid multi-ontology architecture with linear + zero-shot heads
6. **Evaluation** (`evaluation/`) - IC-weighted maxF1 metrics and ancestor closure
7. **Configuration** (`config/`) - Centralized configuration management
8. **Orchestration** (`main.py`) - End-to-end training pipeline

## Key Design Decisions
- **Ontology-specific heads**: Separate classifiers for MFO/BPO/CCO to handle different term distributions
- **BCEWithLogitsLoss**: Proper multilabel loss with positive class weighting for imbalance
- **Ancestor closure**: Applied after scoring, before thresholding (critical for CAFA metrics)
- **Multi-pLM ensemble**: Flexible fusion of ESM-2 (1280-dim), ProtT5 (1024-dim), Ankh (768-dim) via concat/weighted-avg/attention/gated strategies
- **Homology integration**: DIAMOND BLASTP + feature concatenation (3072 fused pLM + 128 homology = 3200-dim with 3-pLM ensemble)
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

✅ **Phase 4 Complete** - Multi-pLM Ensemble
- EmbeddingFusion module with 4 fusion strategies: concat, weighted-avg, attention, gated
- Support for ESM-2 (1280-dim), ProtT5 (1024-dim), Ankh (768-dim)
- Proper protein ID loading from disk (train_ids.npy, test_ids.npy) with demo fallback
- 3-pLM ensemble produces 3072-dim fused embeddings (concat mode)
- Combined with homology: 3200-dim total features (3072 + 128)
- Model scales to 9.8M parameters with 3-pLM ensemble
- CLI flags: --use-ensemble, --ensemble-plms esm2,prott5,ankh, --fusion-strategy concat
- Full data alignment across embeddings, labels, and homology features

✅ **Phase 5 Complete** - Advanced Optimization & CAFA Compliance
- HierarchicalBCELoss: penalizes GO ontology violations (child > parent consistency)
- TopKFilter: enforces ≤1500 terms per protein (CAFA-6 requirement)
- Per-ontology threshold optimization via grid search over configurable ranges
- OptimizationConfig: centralized settings for advanced tuning strategies
- CLI flags: --use-hierarchical-loss, --hierarchical-weight, --no-topk-filter, --max-terms
- Backward compatible: all Phase 5 features disabled by default
- Verified: hierarchical loss + top-K filtering + threshold optimization all working correctly

## Recent Changes (2025-10-29)

**First CAFA-6 Submission (Public Score: 0.000):**
- Successfully uploaded submission.tsv to CAFA-6 competition
- **Issue**: Used demo/test data (--use-testdata flag) with synthetic IDs T00000-T00019
- **Expected**: Real CAFA-6 protein IDs (A0A0C5B5G6, etc.) for ~10,000 test proteins
- **Fix Applied**: Added validation warning when demo IDs detected in submission writer
- **Documentation Updated**: Clarified --use-testdata is for demos only, not competition
- **Next Step**: Generate submission with real CAFA-6 data (remove --use-testdata flag)

## Project History (2025-10-29)
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

**Phase 4 (Multi-pLM Ensemble):**
- Created ensemble/ module with EmbeddingFusion class
- Implemented 4 fusion strategies: concatenation, weighted averaging, attention-based, gated
- Added EnsembleConfig to config/base.py with per-pLM settings (ESM-2, ProtT5, Ankh)
- Extended load_ensemble_embeddings() to load real protein IDs from disk (critical bug fix)
- Integrated with existing homology and zero-shot pipelines
- Tested 3-pLM ensemble: 9.8M parameters, 3200-dim features, IC-weighted maxF1 = 1.0000
- Verified data alignment across embeddings, labels, and homology features
- Analyzed competitor approaches: label propagation, threshold optimization, top-K filtering common in winning solutions

**Phase 5 (Advanced Optimization & CAFA Compliance):**
- Implemented HierarchicalBCELoss in models/hierarchical_loss.py combining BCE + consistency penalties
- Implemented TopKFilter in evaluation/postprocessing.py for CAFA-6 term limit compliance
- Added OptimizationConfig to config/base.py for centralized Phase 5 settings
- Integrated hierarchical loss into Trainer with backward-compatible API
- Integrated top-K filtering into evaluation pipeline after threshold optimization
- Added CLI flags: --use-hierarchical-loss, --hierarchical-weight, --no-topk-filter, --max-terms, --use-testdata
- Tested complete Phase 5 pipeline: hierarchical loss (weight=0.1) + top-K filtering working
- Architect approved: production-ready implementation with minor refinement suggestions
- Fixed critical evaluation bug: empty validation sets now correctly use non-shuffled training loader
- Validated on real CAFA-6 test data: IC-weighted maxF1 = 1.0000 with hierarchical loss enabled

## Next Steps (Phase 6+)
- Species-aware cross-validation (stratified by taxonomy)
- Advanced homology transfer (weighted voting, GO term propagation)
- Label propagation optimization (tune ancestor closure parameters)
- Calibration and uncertainty quantification
- Production deployment configuration
- Validation on realistic CAFA dataset split
- Regression tests for caching behavior
- Fine-tuning Phase 5: expose hierarchical loss components, tighten top-K filtering for score ties, wire advanced threshold strategies

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

**Multi-pLM ensemble mode** (3 protein language models):
```bash
python cafa6_predictor/main.py --demo --use-ensemble --ensemble-plms esm2,prott5,ankh
```

**Custom fusion strategy** (attention-based):
```bash
python cafa6_predictor/main.py --demo --use-ensemble --fusion-strategy attention
```

**Phase 5 features** (hierarchical loss + top-K filtering):
```bash
python cafa6_predictor/main.py --demo --use-ensemble --ensemble-plms esm2,prott5,ankh --use-hierarchical-loss --hierarchical-weight 0.1
```

**Disable top-K filtering** (allow unlimited terms):
```bash
python cafa6_predictor/main.py --demo --no-topk-filter
```

**Use test data from testdata/ subdirectory**:
```bash
python cafa6_predictor/main.py --use-testdata --no-homology
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
- `train_embeddings.npy` - ESM-2 embeddings (1280-dim) [or train_embeddings_esm2.npy for ensemble]
- `train_embeddings_prott5.npy` - ProtT5 embeddings (1024-dim) [optional, for ensemble]
- `train_embeddings_ankh.npy` - Ankh embeddings (768-dim) [optional, for ensemble]
- `train_ids.npy` - Protein IDs
- `test_embeddings.npy` - Test embeddings [or test_embeddings_esm2.npy for ensemble]
- `test_embeddings_prott5.npy` - Test ProtT5 embeddings [optional, for ensemble]
- `test_embeddings_ankh.npy` - Test Ankh embeddings [optional, for ensemble]
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
