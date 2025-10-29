"""
Hierarchical loss regularization for GO ontology consistency

Penalizes predictions that violate the ontology structure:
- If a term is predicted, all ancestors should have higher/equal scores
- Helps maintain true-path rule during training
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class HierarchicalConsistencyLoss(nn.Module):
    """
    Regularization loss that penalizes ontology violations
    
    For each term, we want: parent_score >= child_score
    Violation penalty: max(0, child_score - parent_score)
    """
    
    def __init__(self, ancestor_indices: Dict[int, set], weight: float = 0.1):
        """
        Initialize hierarchical loss
        
        Args:
            ancestor_indices: Dict mapping term index to set of ancestor indices
            weight: Loss weight (lambda) for regularization term
        """
        super().__init__()
        self.ancestor_indices = ancestor_indices
        self.weight = weight
        
        # Pre-compute parent-child pairs for efficient batch processing
        self.child_parent_pairs = []
        for child_idx, parent_indices in ancestor_indices.items():
            for parent_idx in parent_indices:
                self.child_parent_pairs.append((child_idx, parent_idx))
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical consistency loss
        
        Args:
            predictions: Raw logits or probabilities [batch, n_terms]
            
        Returns:
            Scalar loss value
        """
        if not self.child_parent_pairs:
            return torch.tensor(0.0, device=predictions.device)
        
        total_violation = 0.0
        n_violations = 0
        
        for child_idx, parent_idx in self.child_parent_pairs:
            child_scores = predictions[:, child_idx]
            parent_scores = predictions[:, parent_idx]
            
            # Violation: child > parent
            violations = torch.relu(child_scores - parent_scores)
            total_violation += violations.sum()
            n_violations += (violations > 0).sum().item()
        
        # Average violation across batch and pairs
        if len(self.child_parent_pairs) > 0:
            loss = total_violation / (len(self.child_parent_pairs) * predictions.size(0))
        else:
            loss = torch.tensor(0.0, device=predictions.device)
        
        return self.weight * loss


class HierarchicalBCELoss(nn.Module):
    """
    Combined BCE loss with hierarchical consistency regularization
    
    Total loss = BCE(predictions, targets) + λ * H(predictions)
    where H is the hierarchical consistency penalty
    """
    
    def __init__(self, ancestor_indices: Dict[int, set], 
                 pos_weight: torch.Tensor = None,
                 hierarchical_weight: float = 0.1):
        """
        Initialize combined loss
        
        Args:
            ancestor_indices: Dict mapping term index to ancestor indices
            pos_weight: Positive class weights for BCE (handle imbalance)
            hierarchical_weight: Weight for hierarchical consistency term
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        self.hierarchical_loss = HierarchicalConsistencyLoss(
            ancestor_indices, 
            weight=hierarchical_weight
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            logits: Raw model outputs [batch, n_terms]
            targets: Binary labels [batch, n_terms]
            
        Returns:
            Dict with 'total', 'bce', 'hierarchical' losses
        """
        bce = self.bce_loss(logits, targets)
        
        # Apply sigmoid for hierarchical loss (needs probabilities)
        probs = torch.sigmoid(logits)
        hierarchical = self.hierarchical_loss(probs)
        
        total = bce + hierarchical
        
        return {
            'total': total,
            'bce': bce,
            'hierarchical': hierarchical
        }


def create_hierarchical_loss(go_graph, term_to_idx: Dict[str, int], 
                            pos_weights: torch.Tensor = None,
                            hierarchical_weight: float = 0.1) -> HierarchicalBCELoss:
    """
    Factory function to create hierarchical loss from GO graph
    
    Args:
        go_graph: NetworkX DiGraph of GO ontology
        term_to_idx: Dict mapping GO term ID to index
        pos_weights: Positive class weights tensor
        hierarchical_weight: Weight for hierarchical regularization
        
    Returns:
        HierarchicalBCELoss instance
    """
    # Build ancestor indices from GO graph
    ancestor_indices = {}
    
    for term_id, idx in term_to_idx.items():
        if term_id not in go_graph:
            continue
        
        # Get all ancestors (parents, grandparents, etc.)
        ancestors = set()
        try:
            # NetworkX ancestors() gets all nodes reachable from this term
            for ancestor_id in go_graph.predecessors(term_id):
                if ancestor_id in term_to_idx:
                    ancestors.add(term_to_idx[ancestor_id])
        except:
            pass
        
        if ancestors:
            ancestor_indices[idx] = ancestors
    
    return HierarchicalBCELoss(
        ancestor_indices=ancestor_indices,
        pos_weight=pos_weights,
        hierarchical_weight=hierarchical_weight
    )
