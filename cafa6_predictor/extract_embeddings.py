
"""
Extract protein embeddings from FASTA files using ESM-2, ProtT5, and Ankh.

Usage:
    python cafa6_predictor/extract_embeddings.py --input train_sequences.fasta --output-dir data/embeddings --models esm2,prott5,ankh
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from typing import Dict, List, Tuple
import json


class EmbeddingExtractor:
    """Extract embeddings from protein sequences using various pLMs"""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        Initialize embedding extractor
        
        Args:
            model_name: One of 'esm2', 'prott5', 'ankh'
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        print(f"Loading {model_name} model...")
        
        if model_name == 'esm2':
            from transformers import AutoTokenizer, EsmModel
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.embedding_dim = 1280
            
        elif model_name == 'prott5':
            from transformers import T5Tokenizer, T5EncoderModel
            self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
            self.embedding_dim = 1024
            
        elif model_name == 'ankh':
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
            self.model = AutoModel.from_pretrained("ElnaggarLab/ankh-base")
            self.embedding_dim = 768
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(device).eval()
        print(f"✓ Loaded {model_name} (dim={self.embedding_dim})")
    
    @torch.no_grad()
    def extract(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """
        Extract embedding for a single sequence
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        # Truncate if too long
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Handle ProtT5 special formatting (spaces between residues)
        if self.model_name == 'prott5':
            sequence = ' '.join(list(sequence))
        
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        outputs = self.model(**inputs)
        
        # Mean pooling over sequence length (excluding special tokens)
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs[0]
        
        # Mean pool (shape: [1, seq_len, dim] -> [dim])
        embedding = embeddings.mean(dim=1).squeeze(0).cpu().numpy()
        
        return embedding.astype(np.float32)
    
    def extract_batch(self, sequences: List[Tuple[str, str]], 
                     batch_size: int = 8) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for a batch of sequences
        
        Args:
            sequences: List of (protein_id, sequence) tuples
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping protein_id -> embedding
        """
        embeddings = {}
        
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting {self.model_name}"):
            batch = sequences[i:i+batch_size]
            
            for protein_id, seq in batch:
                try:
                    emb = self.extract(seq)
                    embeddings[protein_id] = emb
                except Exception as e:
                    print(f"⚠ Failed to extract {protein_id}: {e}")
                    # Use zero vector as fallback
                    embeddings[protein_id] = np.zeros(self.embedding_dim, dtype=np.float32)
        
        return embeddings


def load_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """Load protein sequences from FASTA file"""
    sequences = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        sequences.append((record.id, str(record.seq)))
    return sequences


def save_embeddings(embeddings: Dict[str, np.ndarray], 
                   output_dir: Path,
                   model_name: str,
                   split: str):
    """
    Save embeddings as .npy arrays
    
    Args:
        embeddings: Dict mapping protein_id -> embedding
        output_dir: Output directory
        model_name: Model name (esm2, prott5, ankh)
        split: Data split (train, test)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by protein ID for consistency
    sorted_ids = sorted(embeddings.keys())
    
    # Stack embeddings into matrix
    emb_matrix = np.stack([embeddings[pid] for pid in sorted_ids], axis=0)
    ids_array = np.array(sorted_ids)
    
    # Save
    if model_name == 'esm2':
        emb_file = output_dir / f"{split}_embeddings.npy"
        ids_file = output_dir / f"{split}_ids.npy"
    else:
        emb_file = output_dir / f"{split}_embeddings_{model_name}.npy"
        ids_file = output_dir / f"{split}_ids.npy"  # IDs are same across models
    
    np.save(emb_file, emb_matrix)
    if not ids_file.exists():
        np.save(ids_file, ids_array)
    
    print(f"✓ Saved {model_name} embeddings: {emb_matrix.shape}")
    print(f"  Files: {emb_file}, {ids_file}")
    
    # Save metadata
    metadata = {
        'model': model_name,
        'split': split,
        'n_proteins': len(sorted_ids),
        'embedding_dim': emb_matrix.shape[1],
        'protein_ids': sorted_ids[:10]  # First 10 for verification
    }
    metadata_file = output_dir / f"{split}_{model_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract protein embeddings from FASTA")
    parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", type=str, default="cafa6_predictor/data", 
                       help="Output directory for embeddings")
    parser.add_argument("--models", type=str, default="esm2", 
                       help="Comma-separated list: esm2,prott5,ankh")
    parser.add_argument("--split", type=str, default="train", 
                       help="Data split name (train/test)")
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Batch size for extraction")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--max-length", type=int, default=1024, 
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {args.device}")
    
    # Load sequences
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {input_path}")
    
    print(f"\nLoading sequences from {input_path}...")
    sequences = load_fasta(input_path)
    print(f"✓ Loaded {len(sequences)} sequences")
    
    # Extract embeddings for each model
    output_dir = Path(args.output_dir)
    models = [m.strip() for m in args.models.split(',')]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Extracting {model_name.upper()} embeddings")
        print(f"{'='*60}")
        
        try:
            extractor = EmbeddingExtractor(model_name, args.device)
            embeddings = extractor.extract_batch(sequences, args.batch_size)
            save_embeddings(embeddings, output_dir, model_name, args.split)
        except Exception as e:
            print(f"⚠ Failed to extract {model_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Models: {', '.join(models)}")


if __name__ == "__main__":
    main()
