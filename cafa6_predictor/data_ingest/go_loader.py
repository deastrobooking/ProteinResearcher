"""
GO Ontology loader and graph management
"""
import networkx as nx
import obonet
from pathlib import Path
from typing import Dict, Set, List
import pickle


class GOGraphLoader:
    """Load and manage GO ontology graph structure"""
    
    ONTOLOGY_ROOTS = {
        'BPO': 'GO:0008150',
        'CCO': 'GO:0005575',
        'MFO': 'GO:0003674'
    }
    
    NAMESPACE_MAP = {
        'biological_process': 'BPO',
        'cellular_component': 'CCO',
        'molecular_function': 'MFO'
    }
    
    def __init__(self, obo_path: Path, cache_dir: Path = None):
        """
        Initialize GO graph loader
        
        Args:
            obo_path: Path to go-basic.obo file
            cache_dir: Optional cache directory for preprocessed graphs
        """
        self.obo_path = obo_path
        self.cache_dir = cache_dir
        self.graph = None
        self.term_to_ontology = {}
        self.term_names = {}
        self.ontology_terms = {'MFO': [], 'BPO': [], 'CCO': []}
        self.term_indices = {}
        self.ancestor_indices = {}
        
    def load(self, use_cache: bool = True) -> 'GOGraphLoader':
        """Load GO graph from OBO file or cache"""
        cache_path = self.cache_dir / "go_graph.pkl" if self.cache_dir else None
        
        if use_cache and cache_path and cache_path.exists():
            print(f"Loading GO graph from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.graph = cached['graph']
                self.term_to_ontology = cached['term_to_ontology']
                self.term_names = cached['term_names']
                self.ontology_terms = cached['ontology_terms']
                self.term_indices = cached['term_indices']
                self.ancestor_indices = cached['ancestor_indices']
            print(f"✓ Loaded {len(self.graph)} GO terms from cache")
            return self
        
        print(f"Loading GO ontology from: {self.obo_path}")
        self._load_obo()
        self._build_indices()
        
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'term_to_ontology': self.term_to_ontology,
                    'term_names': self.term_names,
                    'ontology_terms': self.ontology_terms,
                    'term_indices': self.term_indices,
                    'ancestor_indices': self.ancestor_indices
                }, f)
            print(f"✓ Cached GO graph to {cache_path}")
        
        return self
    
    def _load_obo(self):
        """Load OBO file and build directed graph"""
        graph_multi = obonet.read_obo(str(self.obo_path))
        self.graph = nx.DiGraph()
        
        for node_id, node_data in graph_multi.nodes(data=True):
            if not isinstance(node_id, str) or not node_id.startswith("GO:"):
                continue
            
            namespace = node_data.get('namespace', '')
            ontology = self.NAMESPACE_MAP.get(namespace)
            
            if ontology:
                self.graph.add_node(node_id, namespace=namespace)
                self.term_to_ontology[node_id] = ontology
                self.term_names[node_id] = node_data.get('name', '')
                
                parents = []
                for rel in ('is_a', 'part_of'):
                    for edge in node_data.get(rel, []):
                        parent = edge.split(' ! ')[0] if ' ! ' in edge else edge
                        if parent.startswith("GO:"):
                            parents.append(parent)
                
                for parent in parents:
                    self.graph.add_edge(node_id, parent)
        
        for onto in ['MFO', 'BPO', 'CCO']:
            terms = [t for t, o in self.term_to_ontology.items() if o == onto]
            self.ontology_terms[onto] = sorted(terms)
        
        print(f"✓ Loaded {len(self.graph)} GO terms")
        for onto in ['MFO', 'BPO', 'CCO']:
            print(f"  - {onto}: {len(self.ontology_terms[onto])} terms")
    
    def _build_indices(self):
        """Build term→index mappings and ancestor indices for each ontology"""
        for onto in ['MFO', 'BPO', 'CCO']:
            terms = self.ontology_terms[onto]
            term_idx = {term: i for i, term in enumerate(terms)}
            self.term_indices[onto] = term_idx
            
            ancestor_idx = {}
            for i, term in enumerate(terms):
                ancestors = nx.ancestors(self.graph, term)
                ancestor_idx[i] = {term_idx[a] for a in ancestors if a in term_idx}
            
            self.ancestor_indices[onto] = ancestor_idx
        
        print(f"✓ Built ancestor indices for all ontologies")
    
    def get_ontology(self, term: str) -> str:
        """Get ontology (MFO/BPO/CCO) for a GO term"""
        return self.term_to_ontology.get(term)
    
    def get_ancestors(self, term: str) -> Set[str]:
        """Get all ancestor terms"""
        if term not in self.graph:
            return set()
        return nx.ancestors(self.graph, term)
