# 🎯 CAFA-6 Competition: Next Steps to Improve Your Score

## 📊 Current Status
- ✅ First submission successful (uploaded correctly)
- ❌ Public score: **0.000** 
- 🔍 **Root Cause**: Used demo test data instead of real CAFA-6 data

## 🚨 The Problem
Your submission used **synthetic demo protein IDs** (T00000-T00019) but CAFA-6 expects **~10,000 real protein IDs** (A0A0C5B5G6, A0A1B0GTW7, etc.).

**What happened:**
```bash
# You ran with --use-testdata flag
python cafa6_predictor/main.py --use-testdata --no-homology
# This loaded data from: cafa6_predictor/data/testdata/
# Which contains: 20 demo proteins with IDs T00000-T00019
```

## ✅ Solution: Run with Real CAFA-6 Data

### Step 1: Prepare Real CAFA-6 Data
Place these files in `cafa6_predictor/data/` (NOT in testdata/):

**Required Files:**
- `go-basic.obo` - GO ontology
- `train_terms.tsv` - Training annotations (tab-separated!)
- `IA.tsv` - Information Accretion weights (tab-separated!)
- `train_embeddings.npy` - ESM-2 embeddings for training proteins
- `train_ids.npy` - Training protein IDs
- `test_embeddings.npy` - Test embeddings (~10,000 proteins)
- `test_ids.npy` - Test protein IDs (should match sample_submission.tsv)

**Optional (for better performance):**
- `train_sequences.fasta` - For DIAMOND homology search
- `train_embeddings_prott5.npy` - ProtT5 embeddings (1024-dim)
- `train_embeddings_ankh.npy` - Ankh embeddings (768-dim)

### Step 2: Run Training on Real Data

```bash
# Option 1: Baseline (fastest, single pLM)
python cafa6_predictor/main.py --no-homology

# Option 2: With hierarchical loss (recommended)
python cafa6_predictor/main.py --no-homology \
  --use-hierarchical-loss --hierarchical-weight 0.1

# Option 3: Full ensemble (if you have multi-pLM embeddings)
python cafa6_predictor/main.py \
  --use-ensemble --ensemble-plms esm2,prott5,ankh \
  --use-hierarchical-loss
```

**IMPORTANT**: DO NOT use `--use-testdata` or `--demo` flags!

### Step 3: Verify Before Submitting

The system now includes validation that will warn you:

```
⚠️  WARNING: Detected 20 demo/test protein IDs (starting with 'T0')
   This submission will score 0.000 on CAFA-6 competition!
   Real CAFA-6 expects protein IDs like: A0A0C5B5G6, A0A1B0GTW7, etc.
   To fix: Remove --use-testdata flag and use real CAFA-6 data
```

If you see this warning, **DO NOT upload the submission**!

### Step 4: Upload New Submission

Once training completes:
1. Check `submission.tsv` has ~10,000 unique proteins
2. Verify protein IDs match expected format (NOT T00000-style)
3. Upload to CAFA-6 competition
4. Check public score (should be > 0.000)

## 🎯 Expected Improvements

Based on CAFA-6 competition benchmarks:

**Baseline approach** (single pLM):
- Expected IC-weighted maxF1: **0.3 - 0.5**
- Training time: ~1 hour on 10K proteins (GPU)

**With Phase 5 features** (hierarchical loss + ensemble):
- Expected IC-weighted maxF1: **0.5 - 0.7**
- Training time: ~2-3 hours (GPU recommended)

**Competitive score** (top 20%):
- Target IC-weighted maxF1: **> 0.7**
- Requires: multi-pLM ensemble + homology + tuning

## 🔧 Troubleshooting

**Q: I don't have real CAFA-6 data yet**
- Download from Kaggle competition page
- Extract embeddings using ESM-2/ProtT5/Ankh models
- Or request pre-computed embeddings

**Q: Training takes too long**
- Use GPU: `--device cuda`
- Start with baseline (no ensemble): `--no-homology`
- Reduce epochs in `config/base.py`

**Q: Out of memory**
- Reduce batch size in `config/base.py`
- Use single pLM instead of ensemble
- Process in smaller chunks

**Q: How do I know if I'm using real data?**
- Check test IDs start with real UniProt/protein IDs (A0A..., P00...)
- NOT T00000-style synthetic IDs
- Submission should have ~10,000 proteins, not 20

## 📈 Suggested Next Experiments

1. **Baseline comparison**: Run with/without hierarchical loss
2. **Ensemble ablation**: Test different fusion strategies
3. **Homology impact**: Compare with/without DIAMOND features
4. **Threshold tuning**: Experiment with different grid search ranges

Good luck with your next submission! 🚀
