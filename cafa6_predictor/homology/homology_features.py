import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class HomologyFeatureExtractor:
    def __init__(
        self,
        go_terms: Dict[str, List[str]],
        ontology_terms: Dict[str, List[str]],
        min_identity: float = 30.0,
        min_coverage: float = 0.5,
        top_k_hits: int = 10
    ):
        self.go_terms = go_terms
        self.ontology_terms = ontology_terms
        self.min_identity = min_identity
        self.min_coverage = min_coverage
        self.top_k_hits = top_k_hits
        
    def transfer_go_terms(
        self,
        alignments: pd.DataFrame,
        query_ids: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        transferred_scores = {
            onto: {query_id: np.zeros(len(terms)) 
                   for query_id in query_ids}
            for onto, terms in self.ontology_terms.items()
        }
        
        for query_id in query_ids:
            query_hits = alignments[alignments['qseqid'] == query_id]
            
            query_hits = query_hits[
                (query_hits['pident'] >= self.min_identity)
            ].copy()
            
            if len(query_hits) == 0:
                continue
            
            query_hits = query_hits.nlargest(self.top_k_hits, 'bitscore')
            
            for _, hit in query_hits.iterrows():
                subject_id = hit['sseqid']
                bitscore = hit['bitscore']
                pident = hit['pident']
                
                if subject_id not in self.go_terms:
                    continue
                
                subject_terms = self.go_terms[subject_id]
                
                transfer_weight = (pident / 100.0) * np.log(bitscore + 1) / 10.0
                
                for onto, terms in self.ontology_terms.items():
                    for term in subject_terms:
                        if term in terms:
                            term_idx = terms.index(term)
                            current_score = transferred_scores[onto][query_id][term_idx]
                            transferred_scores[onto][query_id][term_idx] = max(
                                current_score, transfer_weight
                            )
        
        return transferred_scores
    
    def create_homology_features(
        self,
        alignments: pd.DataFrame,
        query_ids: List[str],
        feature_dim: int = 128
    ) -> Dict[str, np.ndarray]:
        features = {}
        
        for query_id in query_ids:
            query_hits = alignments[alignments['qseqid'] == query_id]
            
            if len(query_hits) == 0:
                features[query_id] = np.zeros(feature_dim)
                continue
            
            query_hits = query_hits.nlargest(min(self.top_k_hits, len(query_hits)), 'bitscore')
            
            feat_vector = []
            
            feat_vector.append(len(query_hits))
            feat_vector.append(query_hits['bitscore'].max())
            feat_vector.append(query_hits['bitscore'].mean())
            feat_vector.append(query_hits['pident'].max())
            feat_vector.append(query_hits['pident'].mean())
            feat_vector.append(query_hits['evalue'].min())
            
            top_10_bitscores = query_hits['bitscore'].head(10).tolist()
            top_10_bitscores += [0.0] * (10 - len(top_10_bitscores))
            feat_vector.extend(top_10_bitscores)
            
            top_10_pidents = query_hits['pident'].head(10).tolist()
            top_10_pidents += [0.0] * (10 - len(top_10_pidents))
            feat_vector.extend(top_10_pidents)
            
            while len(feat_vector) < feature_dim:
                feat_vector.append(0.0)
            
            features[query_id] = np.array(feat_vector[:feature_dim])
        
        return features
    
    def combine_features(
        self,
        embeddings: np.ndarray,
        homology_features: Dict[str, np.ndarray],
        protein_ids: List[str]
    ) -> np.ndarray:
        homology_matrix = np.array([
            homology_features.get(pid, np.zeros(128))
            for pid in protein_ids
        ])
        
        homology_matrix = (homology_matrix - homology_matrix.mean(axis=0)) / (homology_matrix.std(axis=0) + 1e-8)
        
        combined = np.concatenate([embeddings, homology_matrix], axis=1)
        
        return combined
