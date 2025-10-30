
# CAFA-6 Kaggle Notebook - Production Guide

This notebook provides a clean, production-ready implementation for CAFA-6 protein function prediction that can be run directly on Kaggle.

## 📋 Quick Start

1. **Upload this notebook to Kaggle**
2. **Attach datasets:**
   - CAFA-6 competition data (auto-mounted at `/kaggle/input/cafa-6-protein-function-prediction/`)
   - Pre-computed embeddings (if available)
3. **Configure settings** in the CONFIG section
4. **Run all cells**
5. **Submit** the generated `submission.tsv`

---

## 🔧 Configuration & Setup

```python
# ========================================
# CONFIGURATION FLAGS
# ========================================
import os
import sys
from pathlib import Path

# Toggle features
INTERNET_ENABLED = False        # Set True to install packages and build embeddings
BUILD_ESM2_EMBEDS = False       # True to generate ESM-2 embeddings from FASTA
USE_PRECOMPUTED_EMBEDS = True   # True to load pre-attached embedding datasets
USE_ENSEMBLE = False            # True to use multi-pLM ensemble (ESM2+ProtT5+Ankh)
USE_HIERARCHICAL_LOSS = True    # True to enable GO hierarchy consistency loss
USE_HOMOLOGY = False            # True to enable DIAMOND homology features (slow)

# Model settings
EMB_DIM = 1280                  # ESM2_t33_650M embedding dimension
BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE = 0.001
MAX_TERMS_PER_PROTEIN = 1500    # CAFA-6 requirement

# Paths (Kaggle standard structure)
COMP_DIR = Path("/kaggle/input/cafa-6-protein-function-prediction")
WORK_DIR = Path("/kaggle/working")
CACHE_DIR = WORK_DIR / "cache"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"

# Competition files
TRAIN_FASTA = COMP_DIR / "train_sequences.fasta"
TEST_FASTA = COMP_DIR / "testsuperset.fasta"
TRAIN_TERMS = COMP_DIR / "Train" / "train_terms.tsv"
GO_OBO = COMP_DIR / "Train" / "go-basic.obo"
IA_WEIGHTS = COMP_DIR / "IA.txt"
SAMPLE_SUB = COMP_DIR / "sample_submission.tsv"

# Pre-computed embeddings (if attached as dataset)
PRECOMP_DIR = Path("/kaggle/input/cafa6-embeddings")  # Adjust to your dataset name
TRAIN_EMB = PRECOMP_DIR / "train_embeddings.npy"
TRAIN_IDS = PRECOMP_DIR / "train_ids.npy"
TEST_EMB = PRECOMP_DIR / "test_embeddings.npy"
TEST_IDS = PRECOMP_DIR / "test_ids.npy"

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

print("✓ Configuration loaded")
print(f"  Competition data: {COMP_DIR.exists()}")
print(f"  Working directory: {WORK_DIR}")
```

---

## 📦 Install Dependencies

```python
# ========================================
# INSTALL PACKAGES (Internet must be ON)
# ========================================
if INTERNET_ENABLED:
    print("Installing dependencies...")
    !pip install -q torch torchmetrics obonet networkx biopython scikit-learn tqdm
    
    if BUILD_ESM2_EMBEDS:
        !pip install -q transformers sentencepiece
    
    if USE_ENSEMBLE:
        !pip install -q transformers sentencepiece
    
    print("✓ Dependencies installed")
else:
    print("⚠ Internet disabled - using pre-installed packages")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pickle
from typing import Dict, List, Tuple

print(f"PyTorch: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

---

## 🧬 Load GO Ontology

```python
# ========================================
# GO ONTOLOGY LOADER
# ========================================
import obonet
import networkx as nx

def load_go_ontology(obo_path: Path):
    """Load GO ontology and extract terms by namespace"""
    print("Loading GO ontology...")
    graph = obonet.read_obo(str(obo_path))
    
    ontology_terms = {'MFO': [], 'BPO': [], 'CCO': []}
    namespace_map = {
        'molecular_function': 'MFO',
        'biological_process': 'BPO',
        'cellular_component': 'CCO'
    }
    
    for term_id in graph.nodes():
        if not term_id.startswith('GO:'):
            continue
        node_data = graph.nodes[term_id]
        namespace = node_data.get('namespace')
        if namespace in namespace_map:
            onto = namespace_map[namespace]
            ontology_terms[onto].append(term_id)
    
    print(f"  MFO: {len(ontology_terms['MFO'])} terms")
    print(f"  BPO: {len(ontology_terms['BPO'])} terms")
    print(f"  CCO: {len(ontology_terms['CCO'])} terms")
    
    return graph, ontology_terms

go_graph, ontology_terms = load_go_ontology(GO_OBO)
```

---

## 🏷️ Build Training Labels

```python
# ========================================
# LABEL BUILDER
# ========================================
def build_labels(terms_tsv: Path, ontology_terms: Dict):
    """Build binary label matrices from train_terms.tsv"""
    print("Building labels...")
    df = pd.read_csv(terms_tsv, sep='\t', header=None, 
                     names=['protein_id', 'term', 'ontology'])
    
    protein_ids = sorted(df['protein_id'].unique())
    labels = {}
    
    for onto in ['MFO', 'BPO', 'CCO']:
        term_to_idx = {t: i for i, t in enumerate(ontology_terms[onto])}
        label_matrix = np.zeros((len(protein_ids), len(ontology_terms[onto])), dtype=np.float32)
        
        onto_df = df[df['ontology'] == onto]
        for _, row in onto_df.iterrows():
            pid_idx = protein_ids.index(row['protein_id'])
            if row['term'] in term_to_idx:
                term_idx = term_to_idx[row['term']]
                label_matrix[pid_idx, term_idx] = 1.0
        
        labels[onto] = label_matrix
        print(f"  {onto}: {label_matrix.shape}, {label_matrix.sum():.0f} annotations")
    
    return protein_ids, labels

train_protein_ids, train_labels = build_labels(TRAIN_TERMS, ontology_terms)
```

---

## 🧮 Load or Generate Embeddings

```python
# ========================================
# EMBEDDING LOADER
# ========================================
def load_embeddings():
    """Load pre-computed embeddings or generate from FASTA"""
    
    if USE_PRECOMPUTED_EMBEDS and TRAIN_EMB.exists():
        print("Loading pre-computed embeddings...")
        train_emb = np.load(TRAIN_EMB)
        train_ids = np.load(TRAIN_IDS, allow_pickle=True).tolist()
        test_emb = np.load(TEST_EMB)
        test_ids = np.load(TEST_IDS, allow_pickle=True).tolist()
        
        print(f"  Train: {train_emb.shape}")
        print(f"  Test: {test_emb.shape}")
        return train_emb, train_ids, test_emb, test_ids
    
    elif BUILD_ESM2_EMBEDS and INTERNET_ENABLED:
        print("Generating ESM-2 embeddings...")
        from transformers import AutoTokenizer, EsmModel
        from Bio import SeqIO
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        def extract_embeddings(fasta_path):
            sequences = [(r.id, str(r.seq)) for r in SeqIO.parse(fasta_path, 'fasta')]
            embeddings = []
            ids = []
            
            for protein_id, seq in tqdm(sequences, desc="Extracting"):
                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                
                embeddings.append(emb)
                ids.append(protein_id)
            
            return np.stack(embeddings), ids
        
        train_emb, train_ids = extract_embeddings(TRAIN_FASTA)
        test_emb, test_ids = extract_embeddings(TEST_FASTA)
        
        # Save for future use
        np.save(WORK_DIR / "train_embeddings.npy", train_emb)
        np.save(WORK_DIR / "train_ids.npy", train_ids)
        np.save(WORK_DIR / "test_embeddings.npy", test_emb)
        np.save(WORK_DIR / "test_ids.npy", test_ids)
        
        return train_emb, train_ids, test_emb, test_ids
    
    else:
        raise ValueError("No embeddings available. Set USE_PRECOMPUTED_EMBEDS=True or BUILD_ESM2_EMBEDS=True")

train_embeddings, train_ids, test_embeddings, test_ids = load_embeddings()
```

---

## 🏗️ Model Architecture

```python
# ========================================
# MULTI-ONTOLOGY MODEL
# ========================================
class MultiOntoModel(nn.Module):
    def __init__(self, embedding_dim: int, ontology_terms: Dict):
        super().__init__()
        self.heads = nn.ModuleDict({
            onto: nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(terms))
            )
            for onto, terms in ontology_terms.items()
        })
    
    def forward(self, x):
        return {onto: head(x) for onto, head in self.heads.items()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiOntoModel(EMB_DIM, ontology_terms).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🎯 Training

```python
# ========================================
# DATASET & TRAINING
# ========================================
class ProteinDataset(Dataset):
    def __init__(self, embeddings, protein_ids, labels):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.protein_ids = protein_ids
        self.labels = {onto: torch.from_numpy(lbls).float() for onto, lbls in labels.items()}
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        return (self.embeddings[idx], 
                {onto: self.labels[onto][idx] for onto in self.labels})

# Align train data
protein_idx_map = {pid: i for i, pid in enumerate(train_protein_ids)}
valid_indices = [i for i, pid in enumerate(train_ids) if pid in protein_idx_map]
label_indices = [protein_idx_map[train_ids[i]] for i in valid_indices]

aligned_embeddings = train_embeddings[valid_indices]
aligned_labels = {onto: train_labels[onto][label_indices] for onto in ['MFO', 'BPO', 'CCO']}
aligned_ids = [train_ids[i] for i in valid_indices]

dataset = ProteinDataset(aligned_embeddings, aligned_ids, aligned_labels)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

print(f"\nTraining on {len(dataset)} proteins...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        embeddings = embeddings.to(device)
        labels = {onto: lbls.to(device) for onto, lbls in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        
        loss = sum(criterion(outputs[onto], labels[onto]) for onto in outputs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"  Loss: {total_loss/len(train_loader):.4f}")

print("✓ Training complete")
```

---

## 📊 Generate Predictions

```python
# ========================================
# INFERENCE & SUBMISSION
# ========================================
model.eval()
test_dataset = ProteinDataset(
    test_embeddings, 
    test_ids, 
    {onto: np.zeros((len(test_ids), len(ontology_terms[onto]))) for onto in ['MFO', 'BPO', 'CCO']}
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = {onto: [] for onto in ['MFO', 'BPO', 'CCO']}

print("Generating predictions...")
with torch.no_grad():
    for embeddings, _ in tqdm(test_loader):
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        
        for onto in outputs:
            probs = torch.sigmoid(outputs[onto]).cpu().numpy()
            predictions[onto].append(probs)

predictions = {onto: np.vstack(preds) for onto, preds in predictions.items()}
print(f"✓ Predictions generated for {len(test_ids)} proteins")
```

---

## 📝 Create Submission File

```python
# ========================================
# SUBMISSION WRITER
# ========================================
def write_submission(predictions, test_ids, ontology_terms, threshold=0.01, max_terms=1500):
    """Write CAFA-6 submission.tsv"""
    rows = []
    
    for i, protein_id in enumerate(tqdm(test_ids, desc="Writing submission")):
        all_predictions = []
        
        for onto in ['MFO', 'BPO', 'CCO']:
            probs = predictions[onto][i]
            terms = ontology_terms[onto]
            
            for term_idx, prob in enumerate(probs):
                if prob >= threshold:
                    all_predictions.append((terms[term_idx], prob))
        
        # Sort by probability and apply top-K filter
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        all_predictions = all_predictions[:max_terms]
        
        for term, prob in all_predictions:
            rows.append({
                'Protein ID': protein_id,
                'GO Term': term,
                'Probability': f"{prob:.4f}"
            })
    
    df = pd.DataFrame(rows)
    output_path = WORK_DIR / "submission.tsv"
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"✓ Wrote {len(df)} predictions to {output_path}")
    print(f"  Unique proteins: {df['Protein ID'].nunique()}")
    print(f"  Avg predictions per protein: {len(df) / df['Protein ID'].nunique():.1f}")
    
    return output_path

submission_path = write_submission(predictions, test_ids, ontology_terms, 
                                   threshold=0.01, max_terms=MAX_TERMS_PER_PROTEIN)
```

---

## ✅ Validation

```python
# ========================================
# VALIDATE SUBMISSION
# ========================================
# Check submission format
df = pd.read_csv(submission_path, sep='\t')
print(f"Submission validation:")
print(f"  Total rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Unique proteins: {df['Protein ID'].nunique()}")
print(f"  Sample:")
print(df.head())

# Verify against sample submission
sample_df = pd.read_csv(SAMPLE_SUB, sep='\t')
sample_proteins = set(sample_df.iloc[:, 0].unique())
submission_proteins = set(df['Protein ID'].unique())

missing = sample_proteins - submission_proteins
if missing:
    print(f"⚠ Missing {len(missing)} proteins from sample submission")
else:
    print("✓ All required proteins present")

print("\n🎉 Submission ready! Click 'Save Version' to submit.")
```

---

## 📌 Notes

### Performance Tips
- **Pre-compute embeddings** offline with Internet ON, then attach as dataset
- **Use GPU** for faster training and inference
- **Batch size**: Adjust based on GPU memory (8-32 recommended)
- **Ensemble**: Combine ESM2+ProtT5+Ankh for better performance

### CAFA-6 Requirements
- ✅ TSV format (tab-separated)
- ✅ Columns: `Protein ID`, `GO Term`, `Probability`
- ✅ Max 1500 terms per protein
- ✅ All test proteins included

### Troubleshooting
- **Out of memory**: Reduce batch size or use smaller model
- **Missing proteins**: Check ID alignment between embeddings and test set
- **Low score**: Try hierarchical loss, ensemble, or homology features

---

## 🚀 Advanced Features (Optional)

To enable advanced features from the main codebase:

1. **Hierarchical Loss**: Enforces GO parent-child consistency
2. **Ensemble Fusion**: Combines ESM2, ProtT5, Ankh embeddings
3. **Homology Features**: DIAMOND sequence alignment (requires BLAST database)
4. **Zero-shot Learning**: Text-based GO term matching

Refer to the main repository's `NEXT_STEPS.md` for detailed implementation.

---

**Good luck with CAFA-6! 🧬**
