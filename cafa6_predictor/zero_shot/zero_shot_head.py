import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroShotHead(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        term_dim: int,
        hidden_dim: int = 512,
        method: str = 'bilinear'
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.term_dim = term_dim
        self.method = method
        
        if method == 'bilinear':
            self.bilinear = nn.Bilinear(protein_dim, term_dim, 1, bias=True)
            
        elif method == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(protein_dim + term_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        elif method == 'cosine':
            self.protein_proj = nn.Linear(protein_dim, term_dim)
            self.temperature = nn.Parameter(torch.tensor(1.0))
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, protein_emb: torch.Tensor, term_embs: torch.Tensor) -> torch.Tensor:
        batch_size = protein_emb.shape[0]
        num_terms = term_embs.shape[0]
        
        if self.method == 'bilinear':
            protein_expanded = protein_emb.unsqueeze(1).expand(-1, num_terms, -1)
            term_expanded = term_embs.unsqueeze(0).expand(batch_size, -1, -1)
            
            logits = self.bilinear(protein_expanded, term_expanded).squeeze(-1)
            
        elif self.method == 'mlp':
            protein_expanded = protein_emb.unsqueeze(1).expand(-1, num_terms, -1)
            term_expanded = term_embs.unsqueeze(0).expand(batch_size, -1, -1)
            
            combined = torch.cat([protein_expanded, term_expanded], dim=-1)
            logits = self.mlp(combined).squeeze(-1)
            
        elif self.method == 'cosine':
            protein_proj = self.protein_proj(protein_emb)
            
            protein_norm = F.normalize(protein_proj, p=2, dim=1)
            term_norm = F.normalize(term_embs, p=2, dim=1)
            
            logits = torch.matmul(protein_norm, term_norm.T) * self.temperature
        
        return logits


class HybridZeroShotHead(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        term_dim: int,
        num_terms: int,
        use_zero_shot: bool = True,
        use_linear: bool = True,
        alpha: float = 0.5
    ):
        super().__init__()
        self.use_zero_shot = use_zero_shot
        self.use_linear = use_linear
        self.alpha = alpha
        
        if use_linear:
            self.linear_head = nn.Linear(protein_dim, num_terms)
        
        if use_zero_shot:
            self.zero_shot_head = ZeroShotHead(protein_dim, term_dim, method='bilinear')
    
    def forward(
        self,
        protein_emb: torch.Tensor,
        term_embs: torch.Tensor = None
    ) -> torch.Tensor:
        logits = None
        
        if self.use_linear:
            linear_logits = self.linear_head(protein_emb)
            logits = linear_logits
        
        if self.use_zero_shot and term_embs is not None:
            zs_logits = self.zero_shot_head(protein_emb, term_embs)
            
            if logits is not None:
                logits = self.alpha * logits + (1 - self.alpha) * zs_logits
            else:
                logits = zs_logits
        
        return logits
