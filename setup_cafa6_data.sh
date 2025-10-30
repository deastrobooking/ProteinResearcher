
#!/bin/bash
# Setup script for CAFA-6 data and embeddings

set -e  # Exit on error

echo "========================================="
echo "CAFA-6 Data Setup Script"
echo "========================================="

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "⚠ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Download CAFA-6 data
echo ""
echo "[Step 1/4] Downloading CAFA-6 competition data..."
mkdir -p cafa6_predictor/data
cd cafa6_predictor/data

if [ ! -f "train_sequences.fasta" ]; then
    kaggle competitions download -c cafa-6-protein-function-prediction
    unzip -o cafa-6-protein-function-prediction.zip
    rm cafa-6-protein-function-prediction.zip
    echo "✓ Downloaded CAFA-6 data"
else
    echo "✓ CAFA-6 data already exists"
fi

cd ../..

# Install embedding extraction dependencies
echo ""
echo "[Step 2/4] Installing dependencies..."
pip install transformers sentencepiece biopython

# Extract ESM-2 embeddings (train)
echo ""
echo "[Step 3/4] Extracting ESM-2 embeddings (this may take a while)..."
if [ ! -f "cafa6_predictor/data/train_embeddings.npy" ]; then
    python cafa6_predictor/extract_embeddings.py \
        --input cafa6_predictor/data/train_sequences.fasta \
        --output-dir cafa6_predictor/data \
        --models esm2 \
        --split train \
        --batch-size 4
else
    echo "✓ Train embeddings already exist"
fi

# Extract ESM-2 embeddings (test)
if [ -f "cafa6_predictor/data/testsuperset.fasta" ] && [ ! -f "cafa6_predictor/data/test_embeddings.npy" ]; then
    python cafa6_predictor/extract_embeddings.py \
        --input cafa6_predictor/data/testsuperset.fasta \
        --output-dir cafa6_predictor/data \
        --models esm2 \
        --split test \
        --batch-size 4
else
    echo "⚠ Test FASTA not found or embeddings already exist"
fi

# Optional: Extract multi-pLM embeddings
echo ""
echo "[Step 4/4] Optional: Extract ProtT5 and Ankh embeddings? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Extracting ProtT5 and Ankh embeddings..."
    python cafa6_predictor/extract_embeddings.py \
        --input cafa6_predictor/data/train_sequences.fasta \
        --output-dir cafa6_predictor/data \
        --models prott5,ankh \
        --split train \
        --batch-size 2
    
    if [ -f "cafa6_predictor/data/testsuperset.fasta" ]; then
        python cafa6_predictor/extract_embeddings.py \
            --input cafa6_predictor/data/testsuperset.fasta \
            --output-dir cafa6_predictor/data \
            --models prott5,ankh \
            --split test \
            --batch-size 2
    fi
else
    echo "Skipping multi-pLM extraction"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "You can now run:"
echo "  python cafa6_predictor/main.py --no-homology"
echo ""
echo "For ensemble mode (if you extracted ProtT5/Ankh):"
echo "  python cafa6_predictor/main.py --no-homology --use-ensemble --ensemble-plms esm2,prott5,ankh"
