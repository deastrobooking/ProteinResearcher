"""
Main training pipeline for CAFA-6 prediction
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import argparse

from config.base import Config
from data_ingest import GOGraphLoader, LabelBuilder, IAWeightLoader
from models import build_model
from models.dataset import CAFA6Dataset
from models.trainer import Trainer
from evaluation import AncestorClosure, evaluate_predictions, SubmissionWriter
from homology import DiamondRunner, HomologyFeatureExtractor


def load_sequences(config: Config):
    """Load protein sequences for homology search"""
    sequences = {}
    
    if config.paths.train_sequences.exists():
        from Bio import SeqIO
        for record in SeqIO.parse(config.paths.train_sequences, 'fasta'):
            sequences[record.id] = str(record.seq)
    else:
        print("⚠ Sequence file not found, generating dummy sequences...")
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        for i in range(100):
            pid = f"P{i:05d}"
            seq_len = np.random.randint(100, 500)
            seq = ''.join(np.random.choice(list(amino_acids), seq_len))
            sequences[pid] = seq
    
    return sequences


def load_embeddings(config: Config):
    """Load pre-computed embeddings"""
    print("\n[Loading Embeddings]")
    
    base_emb_dim = 1280
    
    if not config.paths.train_embeddings.exists():
        print(f"⚠ Training embeddings not found at {config.paths.train_embeddings}")
        print("Creating dummy embeddings for demonstration...")
        
        n_train = 100
        n_test = 20
        
        train_embeddings = np.random.randn(n_train, base_emb_dim).astype(np.float32)
        train_ids = [f"P{i:05d}" for i in range(n_train)]
        test_embeddings = np.random.randn(n_test, base_emb_dim).astype(np.float32)
        test_ids = [f"T{i:05d}" for i in range(n_test)]
        
        config.paths.train_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(config.paths.train_embeddings, train_embeddings)
        np.save(config.paths.train_ids, train_ids)
        np.save(config.paths.test_embeddings, test_embeddings)
        np.save(config.paths.test_ids, test_ids)
    else:
        train_embeddings = np.load(config.paths.train_embeddings)
        train_ids = np.load(config.paths.train_ids, allow_pickle=True).tolist()
        test_embeddings = np.load(config.paths.test_embeddings)
        test_ids = np.load(config.paths.test_ids, allow_pickle=True).tolist()
    
    print(f"✓ Train embeddings: {train_embeddings.shape}")
    print(f"✓ Test embeddings: {test_embeddings.shape}")
    
    return train_embeddings, train_ids, test_embeddings, test_ids


def create_dummy_data(config: Config):
    """Create minimal dummy data files for demonstration"""
    print("\n[Creating Dummy Data Files]")
    
    data_dir = config.paths.base_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not config.paths.go_obo.exists():
        print("Creating minimal GO ontology file...")
        go_content = """format-version: 1.2
data-version: releases/2025-06-01

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0001234
name: test_bp_term
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0005678
name: test_cc_term
namespace: cellular_component
is_a: GO:0005575

[Term]
id: GO:0003456
name: test_mf_term
namespace: molecular_function
is_a: GO:0003674
"""
        config.paths.go_obo.write_text(go_content)
        print(f"✓ Created {config.paths.go_obo}")
    
    if not config.paths.train_terms.exists():
        print("Creating minimal training terms file...")
        terms_content = "P00001\tGO:0001234\tBPO\nP00001\tGO:0003456\tMFO\nP00002\tGO:0005678\tCCO\n"
        config.paths.train_terms.write_text(terms_content)
        print(f"✓ Created {config.paths.train_terms}")
    
    if not config.paths.ia_weights.exists():
        print("Creating minimal IA weights file...")
        ia_content = "GO:0008150\t0.01\nGO:0005575\t0.01\nGO:0003674\t0.01\nGO:0001234\t2.5\nGO:0005678\t3.2\nGO:0003456\t2.8\n"
        config.paths.ia_weights.write_text(ia_content)
        print(f"✓ Created {config.paths.ia_weights}")


def main(args):
    """Main training pipeline"""
    print("="*80)
    print("CAFA-6 PROTEIN FUNCTION PREDICTION - BASELINE TRAINING")
    print("="*80)
    
    config = Config()
    config.model.device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {config.model.device}")
    
    if args.no_homology:
        config.homology.use_homology = False
        config.model.embedding_dim = 1280
        print("⚠ Homology features disabled")
    
    if args.demo:
        create_dummy_data(config)
    
    print("\n[1/7] Loading GO Ontology")
    go_loader = GOGraphLoader(config.paths.go_obo, config.paths.cache_dir)
    go_loader.load(use_cache=not args.no_cache)
    
    print("\n[2/7] Building Labels")
    label_builder = LabelBuilder(go_loader, config.paths.cache_dir)
    label_builder.build_from_tsv(config.paths.train_terms, use_cache=not args.no_cache)
    
    print("\n[3/7] Loading IA Weights")
    ia_loader = IAWeightLoader(go_loader, config.paths.cache_dir)
    ia_loader.load(config.paths.ia_weights, use_cache=not args.no_cache)
    
    print("\n[4/9] Loading Embeddings")
    train_embeddings, train_ids, test_embeddings, test_ids = load_embeddings(config)
    
    if config.homology.use_homology:
        print("\n[5/9] Loading Protein Sequences")
        sequences = load_sequences(config)
        print(f"✓ Loaded {len(sequences)} protein sequences")
        
        print("\n[6/9] Building DIAMOND Database")
        diamond_runner = DiamondRunner(cache_dir=config.paths.cache_dir)
        diamond_runner.build_database(sequences, force_rebuild=args.rebuild_diamond)
        
        print("\n[7/9] Running DIAMOND Homology Search")
        alignments = diamond_runner.run_blastp(
            query_sequences=sequences,
            max_target_seqs=config.homology.max_target_seqs,
            evalue=config.homology.evalue,
            sensitivity=config.homology.sensitivity,
            threads=config.homology.threads
        )
        print(f"✓ Found {len(alignments)} alignments")
        
        print("\n[8/9] Extracting Homology Features")
        protein_go_map = {}
        for pid in label_builder.protein_ids:
            terms = []
            for onto in ['MFO', 'BPO', 'CCO']:
                labels = label_builder.labels[onto]
                idx = list(label_builder.protein_ids).index(pid)
                term_indices = np.where(labels[idx] > 0)[0]
                for term_idx in term_indices:
                    terms.append(go_loader.ontology_terms[onto][term_idx])
            protein_go_map[pid] = terms
        
        homology_extractor = HomologyFeatureExtractor(
            go_terms=protein_go_map,
            ontology_terms=go_loader.ontology_terms,
            min_identity=config.homology.min_identity,
            min_coverage=config.homology.min_coverage,
            top_k_hits=config.homology.top_k_hits
        )
        
        homology_features_train = homology_extractor.create_homology_features(
            alignments, train_ids, config.homology.homology_feature_dim
        )
        
        train_embeddings = homology_extractor.combine_features(
            train_embeddings, homology_features_train, train_ids
        )
        
        print(f"✓ Combined embeddings shape: {train_embeddings.shape}")
    else:
        print("\n[5/9] Skipping homology features (disabled)")
    
    print(f"\n[9/9] Preparing Dataset")
    protein_idx_map = {pid: i for i, pid in enumerate(label_builder.protein_ids)}
    valid_indices = [i for i, pid in enumerate(train_ids) if pid in protein_idx_map]
    
    if len(valid_indices) == 0:
        print("\n⚠ No matching proteins between embeddings and labels!")
        print("Using all embeddings for demonstration...")
        valid_indices = list(range(min(len(train_ids), len(label_builder.protein_ids))))
        aligned_labels = {
            onto: label_builder.labels[onto][:len(valid_indices)]
            for onto in ['MFO', 'BPO', 'CCO']
        }
        aligned_ids = train_ids[:len(valid_indices)]
        aligned_embeddings = train_embeddings[:len(valid_indices)]
    else:
        label_indices = [protein_idx_map[train_ids[i]] for i in valid_indices]
        aligned_labels = {
            onto: label_builder.labels[onto][label_indices]
            for onto in ['MFO', 'BPO', 'CCO']
        }
        aligned_ids = [train_ids[i] for i in valid_indices]
        aligned_embeddings = train_embeddings[valid_indices]
    
    print(f"\n✓ Aligned {len(aligned_ids)} proteins")
    
    dataset = CAFA6Dataset(aligned_embeddings, aligned_ids, aligned_labels)
    
    n_samples = len(dataset)
    val_size = int(n_samples * config.evaluation.validation_split)
    train_size = n_samples - val_size
    
    rng = np.random.RandomState(config.evaluation.random_seed)
    indices = rng.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.model.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.model.batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    print("\nBuilding Model")
    model = build_model(go_loader, config.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model built with {total_params:,} parameters")
    for onto in ['MFO', 'BPO', 'CCO']:
        print(f"  {onto}: {len(go_loader.ontology_terms[onto])} output terms")
    
    print("\nTraining")
    pos_weights = label_builder.compute_pos_weights()
    trainer = Trainer(model, config.model, pos_weights, config.model.device)
    
    best_val_f1 = 0.0
    
    for epoch in range(config.model.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.model.num_epochs}")
        
        train_metrics = trainer.train_epoch(train_loader)
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train F1 - MFO: {train_metrics['MFO_F1']:.3f}, "
              f"BPO: {train_metrics['BPO_F1']:.3f}, CCO: {train_metrics['CCO_F1']:.3f}")
        
        val_mean_f1 = (train_metrics['MFO_F1'] + train_metrics['BPO_F1'] + train_metrics['CCO_F1']) / 3
        
        if val_mean_f1 > best_val_f1:
            best_val_f1 = val_mean_f1
            checkpoint_path = config.paths.checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
    
    print("\nEvaluation")
    
    if len(val_dataset) == 0:
        print("⚠ Validation set is empty (too few samples). Using training set for evaluation.")
        eval_loader = train_loader
        eval_indices = train_indices
    else:
        print("Generating validation predictions...")
        eval_loader = val_loader
        eval_indices = val_indices
    
    eval_predictions = trainer.predict(eval_loader)
    
    eval_labels = {
        onto: aligned_labels[onto][eval_indices]
        for onto in ['MFO', 'BPO', 'CCO']
    }
    
    ancestor_closures = {
        onto: AncestorClosure(go_loader.ancestor_indices[onto])
        for onto in ['MFO', 'BPO', 'CCO']
    }
    
    threshold_range = np.linspace(
        config.evaluation.threshold_min,
        config.evaluation.threshold_max,
        config.evaluation.threshold_search_points
    )
    
    results = evaluate_predictions(
        eval_predictions,
        eval_labels,
        ia_loader.ia_vectors,
        ancestor_closures,
        threshold_range
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Best Mean IC-weighted F1: {results['mean']['max_f1']:.4f}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAFA-6 Baseline Training with Homology Features")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--demo", action="store_true", help="Use demo mode with dummy data")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-homology", action="store_true", help="Disable DIAMOND homology features")
    parser.add_argument("--rebuild-diamond", action="store_true", help="Force rebuild DIAMOND database")
    
    args = parser.parse_args()
    main(args)
