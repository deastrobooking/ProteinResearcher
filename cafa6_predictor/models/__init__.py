"""
Model architectures for CAFA-6 prediction
"""

from .multionto_model import MultiOntoModel, build_model
from .hybrid_model import HybridMultiOntoModel, build_hybrid_model
from .hierarchical_loss import HierarchicalConsistencyLoss, HierarchicalBCELoss, create_hierarchical_loss

__all__ = [
    'MultiOntoModel', 
    'build_model', 
    'HybridMultiOntoModel', 
    'build_hybrid_model',
    'HierarchicalConsistencyLoss',
    'HierarchicalBCELoss',
    'create_hierarchical_loss'
]
