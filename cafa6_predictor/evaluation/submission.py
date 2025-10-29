"""
CAFA-6 compliant submission file generation
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class SubmissionWriter:
    """Generate CAFA-6 compliant submission files"""
    
    @staticmethod
    def to_sig3(value: float) -> str:
        """
        Format score to ≤3 significant figures
        
        Args:
            value: Score value
            
        Returns:
            Formatted string
        """
        value = float(np.clip(value, 1e-6, 1.0))
        return f"{value:.3g}"
    
    @staticmethod
    def write_submission(protein_ids: List[str],
                        predictions: Dict[str, np.ndarray],
                        go_terms: Dict[str, List[str]],
                        thresholds: Dict[str, float],
                        ancestor_closures: Dict,
                        output_path: Path,
                        max_terms_per_protein: int = 1500) -> pd.DataFrame:
        """
        Write CAFA-6 compliant submission file
        
        Args:
            protein_ids: List of protein IDs
            predictions: Dict mapping ontology to score matrices [n_proteins, n_terms]
            go_terms: Dict mapping ontology to list of GO term IDs
            thresholds: Dict mapping ontology to classification threshold
            ancestor_closures: Dict mapping ontology to AncestorClosure instances
            output_path: Output file path
            max_terms_per_protein: Maximum terms per protein (default 1500)
            
        Returns:
            DataFrame with submission data
        """
        from .metrics import AncestorClosure
        
        rows = []
        
        for onto in ['MFO', 'BPO', 'CCO']:
            closed_scores = ancestor_closures[onto].apply(predictions[onto])
            
            for i, protein_id in enumerate(protein_ids):
                threshold = thresholds[onto]
                selected_indices = np.where(closed_scores[i] >= threshold)[0]
                
                for j in selected_indices:
                    go_term = go_terms[onto][j]
                    score = float(closed_scores[i, j])
                    rows.append((protein_id, go_term, score))
        
        rows.sort(key=lambda x: (x[0], -x[2]))
        
        filtered_rows = []
        current_protein = None
        count = 0
        
        for protein_id, go_term, score in rows:
            if protein_id != current_protein:
                current_protein = protein_id
                count = 0
            
            if count < max_terms_per_protein:
                filtered_rows.append((protein_id, go_term, SubmissionWriter.to_sig3(score)))
                count += 1
        
        df = pd.DataFrame(filtered_rows, columns=['protein_id', 'go_term', 'confidence'])
        
        df.to_csv(output_path, sep='\t', header=False, index=False)
        
        print(f"✓ Wrote submission to {output_path}")
        print(f"  Total predictions: {len(df):,}")
        print(f"  Unique proteins: {df['protein_id'].nunique():,}")
        print(f"  Predictions per protein: {len(df) / df['protein_id'].nunique():.1f} (avg)")
        
        return df
