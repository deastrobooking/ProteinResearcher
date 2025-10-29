"""
IC-weighted evaluation metrics with ancestor closure
"""
import numpy as np
from typing import Dict, Tuple, List
import torch


class AncestorClosure:
    """Apply ancestor closure to prediction scores"""
    
    def __init__(self, ancestor_indices: Dict[int, set]):
        """
        Initialize ancestor closure
        
        Args:
            ancestor_indices: Dict mapping term index to set of ancestor indices
        """
        self.ancestor_indices = ancestor_indices
    
    def apply(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply ancestor closure: propagate child scores to parents via max
        
        Args:
            scores: Score matrix [n_samples, n_terms]
            
        Returns:
            Closed scores [n_samples, n_terms]
        """
        closed = scores.copy()
        
        changed = True
        max_iterations = 20
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for child_idx, parent_indices in self.ancestor_indices.items():
                if not parent_indices:
                    continue
                
                parent_list = list(parent_indices)
                child_scores = closed[:, child_idx:child_idx+1]
                parent_scores = closed[:, parent_list]
                
                new_parent_scores = np.maximum(parent_scores, child_scores)
                
                if not np.array_equal(new_parent_scores, parent_scores):
                    closed[:, parent_list] = new_parent_scores
                    changed = True
        
        return closed


class ICWeightedMaxF1:
    """IC-weighted F1 score computation with threshold optimization"""
    
    def __init__(self, ia_vector: np.ndarray, ancestor_closure: AncestorClosure):
        """
        Initialize IC-weighted F1 metric
        
        Args:
            ia_vector: IA weights for each term [n_terms]
            ancestor_closure: AncestorClosure instance
        """
        self.ia_vector = ia_vector.reshape(1, -1)
        self.ancestor_closure = ancestor_closure
    
    def compute_f1_at_threshold(self, scores: np.ndarray, y_true: np.ndarray, 
                                threshold: float) -> float:
        """
        Compute IC-weighted F1 at a specific threshold
        
        Args:
            scores: Prediction scores [n_samples, n_terms]
            y_true: Ground truth labels [n_samples, n_terms]
            threshold: Classification threshold
            
        Returns:
            IC-weighted F1 score
        """
        predictions = (scores >= threshold).astype(np.uint8)
        
        tp = ((predictions == 1) & (y_true == 1)) * self.ia_vector
        fp = ((predictions == 1) & (y_true == 0)) * self.ia_vector
        fn = ((predictions == 0) & (y_true == 1)) * self.ia_vector
        
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        
        precision = tp_sum / (tp_sum + fp_sum + 1e-12)
        recall = tp_sum / (tp_sum + fn_sum + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        
        return float(f1)
    
    def find_best_threshold(self, raw_scores: np.ndarray, y_true: np.ndarray,
                           threshold_range: np.ndarray) -> Tuple[float, float]:
        """
        Find optimal threshold that maximizes IC-weighted F1
        
        Args:
            raw_scores: Raw prediction scores [n_samples, n_terms]
            y_true: Ground truth labels [n_samples, n_terms]
            threshold_range: Array of thresholds to search
            
        Returns:
            (best_f1, best_threshold)
        """
        closed_scores = self.ancestor_closure.apply(raw_scores)
        
        best_f1 = 0.0
        best_threshold = 0.5
        
        for threshold in threshold_range:
            f1 = self.compute_f1_at_threshold(closed_scores, y_true, threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_f1, best_threshold


def evaluate_predictions(predictions: Dict[str, np.ndarray], 
                        ground_truth: Dict[str, np.ndarray],
                        ia_vectors: Dict[str, np.ndarray],
                        ancestor_closures: Dict[str, AncestorClosure],
                        threshold_range: np.ndarray) -> Dict[str, Dict]:
    """
    Evaluate predictions across all ontologies
    
    Args:
        predictions: Dict mapping ontology to prediction scores
        ground_truth: Dict mapping ontology to ground truth labels
        ia_vectors: Dict mapping ontology to IA weight vectors
        ancestor_closures: Dict mapping ontology to AncestorClosure instances
        threshold_range: Array of thresholds to search
        
    Returns:
        Dict with evaluation results per ontology
    """
    results = {}
    
    for onto in ['MFO', 'BPO', 'CCO']:
        metric = ICWeightedMaxF1(ia_vectors[onto], ancestor_closures[onto])
        best_f1, best_threshold = metric.find_best_threshold(
            predictions[onto], 
            ground_truth[onto],
            threshold_range
        )
        
        results[onto] = {
            'max_f1': best_f1,
            'best_threshold': best_threshold
        }
        
        print(f"{onto}: IC-weighted maxF1 = {best_f1:.4f} @ threshold = {best_threshold:.3f}")
    
    mean_f1 = np.mean([results[onto]['max_f1'] for onto in ['MFO', 'BPO', 'CCO']])
    results['mean'] = {'max_f1': mean_f1}
    print(f"Mean IC-weighted maxF1 = {mean_f1:.4f}")
    
    return results
