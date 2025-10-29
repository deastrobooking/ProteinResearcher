"""
Training loop for CAFA-6 model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelF1Score
from typing import Dict
import numpy as np
from tqdm import tqdm


class Trainer:
    """Trainer for multi-ontology model"""
    
    def __init__(self, model: nn.Module, config, pos_weights: Dict[str, np.ndarray], device: str = 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: MultiOntoModel instance
            config: Training configuration
            pos_weights: Dict mapping ontology to positive class weights
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.criterions = {
            onto: nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weights[onto], dtype=torch.float32).to(device)
            )
            for onto in ['MFO', 'BPO', 'CCO']
        }
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.metrics = {
            onto: MultilabelF1Score(
                num_labels=model.ontology_sizes[onto],
                average='micro',
                threshold=0.5
            ).to(device)
            for onto in ['MFO', 'BPO', 'CCO']
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dict with loss and F1 metrics
        """
        self.model.train()
        
        total_loss = 0.0
        onto_f1 = {'MFO': 0.0, 'BPO': 0.0, 'CCO': 0.0}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for embeddings, labels in pbar:
            embeddings = embeddings.to(self.device)
            labels = {onto: labels[onto].to(self.device) for onto in ['MFO', 'BPO', 'CCO']}
            
            self.optimizer.zero_grad()
            
            outputs = self.model(embeddings)
            
            loss = sum(
                self.criterions[onto](outputs[onto], labels[onto])
                for onto in ['MFO', 'BPO', 'CCO']
            )
            
            loss.backward()
            
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            for onto in ['MFO', 'BPO', 'CCO']:
                f1 = self.metrics[onto](outputs[onto], labels[onto].int())
                onto_f1[onto] += f1.item()
            
            n_batches += 1
            
            pbar.set_postfix({
                'loss': total_loss / n_batches,
                'MFO_F1': onto_f1['MFO'] / n_batches,
                'BPO_F1': onto_f1['BPO'] / n_batches,
                'CCO_F1': onto_f1['CCO'] / n_batches
            })
        
        return {
            'loss': total_loss / n_batches,
            'MFO_F1': onto_f1['MFO'] / n_batches,
            'BPO_F1': onto_f1['BPO'] / n_batches,
            'CCO_F1': onto_f1['CCO'] / n_batches
        }
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Generate predictions
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dict mapping ontology to prediction scores
        """
        self.model.eval()
        
        all_predictions = {onto: [] for onto in ['MFO', 'BPO', 'CCO']}
        
        for batch in tqdm(data_loader, desc='Predicting', leave=False):
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                embeddings = batch[0]
            elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                embeddings = batch[0]
            else:
                embeddings = batch
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.to(self.device)
            else:
                embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            
            outputs = self.model(embeddings)
            
            for onto in ['MFO', 'BPO', 'CCO']:
                probs = torch.sigmoid(outputs[onto]).cpu().numpy()
                all_predictions[onto].append(probs)
        
        predictions = {
            onto: np.vstack(all_predictions[onto])
            for onto in ['MFO', 'BPO', 'CCO']
        }
        
        return predictions
    
    def save_checkpoint(self, path: str, epoch: int, metrics: dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
