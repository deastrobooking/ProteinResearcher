"""
Model architectures for CAFA-6 prediction
"""

from .multionto_model import MultiOntoModel, build_model
from .hybrid_model import HybridMultiOntoModel, build_hybrid_model

__all__ = ['MultiOntoModel', 'build_model', 'HybridMultiOntoModel', 'build_hybrid_model']
