"""
Data ingestion and preprocessing for CAFA-6
"""

from .go_loader import GOGraphLoader
from .label_builder import LabelBuilder
from .ia_weights import IAWeightLoader

__all__ = ['GOGraphLoader', 'LabelBuilder', 'IAWeightLoader']
