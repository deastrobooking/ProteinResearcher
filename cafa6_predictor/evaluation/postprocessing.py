"""
Submission postprocessing: top-K filtering and threshold optimization
"""
import numpy as np
from typing import Dict, Tuple, List


class TopKFilter:
    """Filter predictions to top-K most confident terms per protein"""
    
    def __init__(self, max_terms_per_protein: int = 1500):
        """
        Initialize top-K filter
        
        Args:
            max_terms_per_protein: Maximum terms to keep per protein (CAFA-6: ≤1500)
        """
        self.max_terms = max_terms_per_protein
    
    def apply(self, scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Filter to top-K highest scoring terms per protein
        
        Args:
            scores: Prediction scores [n_samples, n_terms]
            threshold: Minimum score threshold (terms below this are excluded first)
            
        Returns:
            Filtered scores [n_samples, n_terms] with at most K terms per protein
        """
        filtered = scores.copy()
        n_samples, n_terms = filtered.shape
        
        for i in range(n_samples):
            row_scores = filtered[i]
            
            # First apply threshold
            above_threshold = row_scores >= threshold
            valid_scores = row_scores * above_threshold
            
            # Count non-zero predictions
            n_predictions = np.count_nonzero(valid_scores)
            
            if n_predictions > self.max_terms:
                # Find the K-th highest score
                top_k_threshold = np.partition(valid_scores, -self.max_terms)[-self.max_terms]
                
                # Keep only top-K
                mask = valid_scores >= top_k_threshold
                filtered[i] = row_scores * mask
            else:
                # Keep all above threshold
                filtered[i] = valid_scores
        
        return filtered


class AdaptiveThresholdOptimizer:
    """
    Advanced threshold optimization with multiple strategies
    
    Strategies:
    - global: Single threshold across all proteins
    - per_ontology: One threshold per ontology (default, already in metrics.py)
    - percentile: Adaptive per-protein thresholds based on score distribution
    """
    
    def __init__(self, strategy: str = 'per_ontology'):
        """
        Initialize optimizer
        
        Args:
            strategy: Threshold strategy ('global', 'per_ontology', 'percentile')
        """
        if strategy not in ['global', 'per_ontology', 'percentile']:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
    
    def optimize_percentile_threshold(self, scores: np.ndarray, 
                                     y_true: np.ndarray,
                                     ia_vector: np.ndarray,
                                     percentiles: List[float]) -> Tuple[float, float]:
        """
        Find best percentile for adaptive per-protein thresholding
        
        Args:
            scores: Prediction scores [n_samples, n_terms]
            y_true: Ground truth labels [n_samples, n_terms]
            ia_vector: IA weights [n_terms]
            percentiles: List of percentiles to try (e.g., [50, 60, 70, 80, 90, 95])
            
        Returns:
            (best_f1, best_percentile)
        """
        ia_vector = ia_vector.reshape(1, -1)
        best_f1 = 0.0
        best_percentile = 50.0
        
        for p in percentiles:
            # Apply per-protein percentile threshold
            predictions = np.zeros_like(scores, dtype=np.uint8)
            for i in range(scores.shape[0]):
                if scores[i].max() > 0:
                    threshold = np.percentile(scores[i][scores[i] > 0], p)
                    predictions[i] = (scores[i] >= threshold).astype(np.uint8)
            
            # Compute F1
            tp = ((predictions == 1) & (y_true == 1)) * ia_vector
            fp = ((predictions == 1) & (y_true == 0)) * ia_vector
            fn = ((predictions == 0) & (y_true == 1)) * ia_vector
            
            tp_sum = tp.sum()
            fp_sum = fp.sum()
            fn_sum = fn.sum()
            
            precision = tp_sum / (tp_sum + fp_sum + 1e-12)
            recall = tp_sum / (tp_sum + fn_sum + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            
            if f1 > best_f1:
                best_f1 = f1
                best_percentile = p
        
        return best_f1, best_percentile


def apply_cafa_submission_filters(predictions: Dict[str, np.ndarray],
                                   thresholds: Dict[str, float],
                                   max_terms_per_protein: int = 1500,
                                   use_topk: bool = True) -> Dict[str, np.ndarray]:
    """
    Apply CAFA-6 compliant filtering to predictions
    
    Args:
        predictions: Dict mapping ontology to prediction scores
        thresholds: Dict mapping ontology to optimal threshold
        max_terms_per_protein: Maximum terms per protein (CAFA-6: ≤1500)
        use_topk: Whether to apply top-K filtering
        
    Returns:
        Filtered predictions ready for submission
    """
    filtered = {}
    topk_filter = TopKFilter(max_terms_per_protein)
    
    for onto in ['MFO', 'BPO', 'CCO']:
        scores = predictions[onto]
        threshold = thresholds.get(onto, 0.5)
        
        if use_topk:
            # Apply top-K filtering with threshold
            filtered[onto] = topk_filter.apply(scores, threshold)
        else:
            # Just apply threshold
            filtered[onto] = scores * (scores >= threshold)
    
    return filtered


def count_predictions_per_protein(predictions: Dict[str, np.ndarray]) -> Dict:
    """
    Count predictions per protein across ontologies
    
    Args:
        predictions: Dict mapping ontology to binary predictions
        
    Returns:
        Statistics dict with counts per protein
    """
    stats = {
        'per_protein_counts': [],
        'max_count': 0,
        'mean_count': 0.0,
        'exceeds_limit': 0
    }
    
    # Combine all ontologies
    all_preds = np.concatenate([predictions[o] for o in ['MFO', 'BPO', 'CCO']], axis=1)
    
    for i in range(all_preds.shape[0]):
        count = np.count_nonzero(all_preds[i])
        stats['per_protein_counts'].append(count)
        
        if count > 1500:
            stats['exceeds_limit'] += 1
    
    stats['max_count'] = max(stats['per_protein_counts']) if stats['per_protein_counts'] else 0
    stats['mean_count'] = np.mean(stats['per_protein_counts']) if stats['per_protein_counts'] else 0.0
    
    return stats
