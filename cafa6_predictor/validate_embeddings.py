
"""
Validate extracted embeddings for CAFA-6.

Checks:
- File existence
- Dimension correctness
- ID alignment across models
- No NaN/Inf values
"""

import numpy as np
from pathlib import Path
import json


def validate_embeddings(data_dir: Path = Path("cafa6_predictor/data")):
    """Validate all embeddings in data directory"""
    
    print("="*60)
    print("CAFA-6 Embedding Validation")
    print("="*60)
    
    issues = []
    
    # Expected dimensions
    expected_dims = {
        'esm2': 1280,
        'prott5': 1024,
        'ankh': 768
    }
    
    # Check each split
    for split in ['train', 'test']:
        print(f"\n[{split.upper()}]")
        
        # Check IDs file
        ids_file = data_dir / f"{split}_ids.npy"
        if not ids_file.exists():
            issues.append(f"Missing {split}_ids.npy")
            print(f"  ✗ IDs file missing")
            continue
        
        ids = np.load(ids_file, allow_pickle=True)
        print(f"  ✓ IDs: {len(ids)} proteins")
        
        # Check each model
        for model, expected_dim in expected_dims.items():
            if model == 'esm2':
                emb_file = data_dir / f"{split}_embeddings.npy"
            else:
                emb_file = data_dir / f"{split}_embeddings_{model}.npy"
            
            if not emb_file.exists():
                print(f"  ⚠ {model}: not found (optional)")
                continue
            
            # Load embeddings
            try:
                emb = np.load(emb_file)
            except Exception as e:
                issues.append(f"{split}/{model}: Failed to load - {e}")
                print(f"  ✗ {model}: FAILED TO LOAD")
                continue
            
            # Check shape
            if emb.shape[0] != len(ids):
                issues.append(f"{split}/{model}: Shape mismatch (emb={emb.shape[0]}, ids={len(ids)})")
                print(f"  ✗ {model}: shape mismatch")
                continue
            
            if emb.shape[1] != expected_dim:
                issues.append(f"{split}/{model}: Wrong dimension (got={emb.shape[1]}, expected={expected_dim})")
                print(f"  ✗ {model}: wrong dimension")
                continue
            
            # Check for NaN/Inf
            if np.isnan(emb).any():
                issues.append(f"{split}/{model}: Contains NaN values")
                print(f"  ✗ {model}: contains NaN")
                continue
            
            if np.isinf(emb).any():
                issues.append(f"{split}/{model}: Contains Inf values")
                print(f"  ✗ {model}: contains Inf")
                continue
            
            print(f"  ✓ {model}: {emb.shape} - OK")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print("VALIDATION FAILED")
        print(f"{len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("VALIDATION PASSED")
        print("All embeddings are valid!")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate CAFA-6 embeddings")
    parser.add_argument("--data-dir", type=str, default="cafa6_predictor/data",
                       help="Data directory containing embeddings")
    args = parser.parse_args()
    
    validate_embeddings(Path(args.data_dir))
