import torch
import torch.nn as nn
from typing import Dict, Optional
from zero_shot.zero_shot_head import ZeroShotHead, HybridZeroShotHead


class HybridHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        term_emb_dim: int = None,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        use_zero_shot: bool = False,
        use_linear: bool = True,
        zero_shot_method: str = 'bilinear',
        alpha: float = 0.5
    ):
        super().__init__()
        self.use_zero_shot = use_zero_shot
        self.use_linear = use_linear
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if use_zero_shot and use_linear:
            self.head = HybridZeroShotHead(
                protein_dim=hidden_dim,
                term_dim=term_emb_dim,
                num_terms=out_dim,
                use_zero_shot=True,
                use_linear=True,
                alpha=alpha
            )
        elif use_zero_shot:
            self.head = ZeroShotHead(
                protein_dim=hidden_dim,
                term_dim=term_emb_dim,
                method=zero_shot_method
            )
        else:
            self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        term_embs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = self.feature_extractor(x)
        
        if self.use_zero_shot and term_embs is not None:
            if isinstance(self.head, (ZeroShotHead, HybridZeroShotHead)):
                return self.head(features, term_embs)
            else:
                return self.head(features)
        else:
            return self.head(features)


class HybridMultiOntoModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ontology_sizes: Dict[str, int],
        term_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        use_zero_shot: bool = False,
        use_hybrid: bool = False,
        zero_shot_method: str = 'bilinear',
        alpha: float = 0.5
    ):
        super().__init__()
        self.use_zero_shot = use_zero_shot
        self.use_hybrid = use_hybrid
        
        term_emb_dim = None
        if term_embeddings is not None:
            term_emb_dim = next(iter(term_embeddings.values())).shape[1]
        
        self.mfo = HybridHead(
            in_dim=embedding_dim,
            out_dim=ontology_sizes['MFO'],
            term_emb_dim=term_emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_zero_shot=use_zero_shot or use_hybrid,
            use_linear=not use_zero_shot or use_hybrid,
            zero_shot_method=zero_shot_method,
            alpha=alpha
        )
        
        self.bpo = HybridHead(
            in_dim=embedding_dim,
            out_dim=ontology_sizes['BPO'],
            term_emb_dim=term_emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_zero_shot=use_zero_shot or use_hybrid,
            use_linear=not use_zero_shot or use_hybrid,
            zero_shot_method=zero_shot_method,
            alpha=alpha
        )
        
        self.cco = HybridHead(
            in_dim=embedding_dim,
            out_dim=ontology_sizes['CCO'],
            term_emb_dim=term_emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_zero_shot=use_zero_shot or use_hybrid,
            use_linear=not use_zero_shot or use_hybrid,
            zero_shot_method=zero_shot_method,
            alpha=alpha
        )
        
        if term_embeddings is not None:
            self.register_buffer('mfo_term_embs', term_embeddings['MFO'])
            self.register_buffer('bpo_term_embs', term_embeddings['BPO'])
            self.register_buffer('cco_term_embs', term_embeddings['CCO'])
        else:
            self.mfo_term_embs = None
            self.bpo_term_embs = None
            self.cco_term_embs = None
        
        self.ontology_sizes = ontology_sizes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'MFO': self.mfo(x, self.mfo_term_embs),
            'BPO': self.bpo(x, self.bpo_term_embs),
            'CCO': self.cco(x, self.cco_term_embs)
        }


def build_hybrid_model(
    go_loader,
    config,
    term_embeddings: Optional[Dict[str, torch.Tensor]] = None
) -> nn.Module:
    ontology_sizes = {
        onto: len(go_loader.ontology_terms[onto])
        for onto in ['MFO', 'BPO', 'CCO']
    }
    
    if config.zero_shot.use_zero_shot or config.zero_shot.use_hybrid:
        model = HybridMultiOntoModel(
            embedding_dim=config.model.embedding_dim,
            ontology_sizes=ontology_sizes,
            term_embeddings=term_embeddings,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            use_zero_shot=config.zero_shot.use_zero_shot and not config.zero_shot.use_hybrid,
            use_hybrid=config.zero_shot.use_hybrid,
            zero_shot_method=config.zero_shot.method,
            alpha=config.zero_shot.hybrid_alpha
        )
    else:
        from models.multionto_model import MultiOntoModel
        model = MultiOntoModel(
            embedding_dim=config.model.embedding_dim,
            ontology_sizes=ontology_sizes,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout
        )
    
    return model
