import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import pickle
from typing import List, Dict
import obonet
import networkx as nx


class GOTermEncoder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Path = None,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else Path('cafa6_predictor/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading sentence transformer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        self.term_embeddings = {}
    
    def load_go_descriptions(self, obo_path: Path) -> Dict[str, str]:
        graph = obonet.read_obo(obo_path)
        
        descriptions = {}
        for term_id in graph.nodes():
            node_data = graph.nodes[term_id]
            
            name = node_data.get('name', '')
            definition = node_data.get('def', '')
            
            if definition:
                definition = definition.split('"')[1] if '"' in definition else definition
            
            desc = f"{name}. {definition}".strip() if definition else name
            descriptions[term_id] = desc
        
        return descriptions
    
    @torch.no_grad()
    def encode_terms(
        self,
        term_ids: List[str],
        descriptions: Dict[str, str],
        use_cache: bool = True
    ) -> torch.Tensor:
        cache_path = self.cache_dir / f'term_embeddings_{self.model_name.replace("/", "_")}.pkl'
        
        if use_cache and cache_path.exists():
            print(f"Loading cached term embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            if all(term_id in cached for term_id in term_ids):
                embeddings = torch.stack([cached[term_id] for term_id in term_ids])
                self.term_embeddings = cached
                return embeddings
        
        print(f"Encoding {len(term_ids)} GO terms...")
        texts = [descriptions.get(term_id, term_id) for term_id in term_ids]
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        for term_id, emb in zip(term_ids, all_embeddings):
            self.term_embeddings[term_id] = emb
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.term_embeddings, f)
        print(f"✓ Cached term embeddings to {cache_path}")
        
        return all_embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_ontology_terms(
        self,
        go_loader,
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        descriptions = self.load_go_descriptions(go_loader.obo_path)
        
        ontology_embeddings = {}
        for onto in ['MFO', 'BPO', 'CCO']:
            term_ids = go_loader.ontology_terms[onto]
            embeddings = self.encode_terms(term_ids, descriptions, use_cache)
            ontology_embeddings[onto] = embeddings
            print(f"✓ {onto}: {len(term_ids)} terms encoded, embedding dim: {embeddings.shape[1]}")
        
        return ontology_embeddings
