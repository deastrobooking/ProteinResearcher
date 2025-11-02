"""
Professional ML Competition Features for CAFA-6 Pipeline
========================================================

This module implements 6 professional-grade ML competition features:
1. Smart Dataset Auto-Detection
2. Bayesian Hyperparameter Optimization
3. Feature/Operation Importance Tracking
4. Multi-Model Ensemble System
5. Performance Monitoring Dashboard
6. Advanced Submission Generation

Optimized for protein function prediction and CAFA-6 competition.
"""

import os
import sys
import time
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Callable, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

# Optional dependencies (with fallbacks)
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    warnings.warn("sklearn not available - Bayesian optimization disabled")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - neural network features disabled")

# =============================================================================
# Feature 1: Smart Dataset Auto-Detection
# =============================================================================

class DatasetDetector:
    """Auto-detect CAFA-6 datasets across multiple locations."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.dataset_paths = {}
        self.validation_results = {}
        
    def scan_directories(self, search_paths: List[str] = None) -> Dict[str, Path]:
        """Scan multiple directories for CAFA-6 datasets."""
        if search_paths is None:
            search_paths = self._default_search_paths()
        
        found_datasets = {}
        
        if self.verbose:
            print("\n📁 SCANNING FOR CAFA-6 DATASETS")
            print("="*50)
        
        for base_path in search_paths:
            if self.verbose:
                print(f"\n🔍 Searching: {base_path}")
            
            datasets = self._find_cafa6_datasets(Path(base_path))
            found_datasets.update(datasets)
            
            if datasets and self.verbose:
                for name, path in datasets.items():
                    print(f"  ✓ Found {name}: {path}")
        
        self.dataset_paths = found_datasets
        return found_datasets
    
    def _default_search_paths(self) -> List[str]:
        """Default search paths for different environments."""
        paths = [
            "/kaggle/input",           # Kaggle environment
            "../input",                # Local Kaggle-style
            "./data",                  # Standard ML structure
            "./cafa6_predictor/data",  # Our project structure
            "~/data",                  # User home
            "/content/data",           # Colab
        ]
        
        # Only return paths that exist
        return [p for p in paths if Path(p).exists()]
    
    def _find_cafa6_datasets(self, path: Path) -> Dict[str, Path]:
        """Find CAFA-6 specific datasets."""
        datasets = {}
        
        if not path.exists():
            return datasets
        
        # CAFA-6 specific file patterns
        patterns = {
            'go_ontology': ['go-basic.obo', 'go.obo'],
            'train_terms': ['train_terms.tsv', 'training_terms.tsv'],
            'ia_weights': ['IA.tsv', 'information_accretion.tsv'],
            'train_embeddings': ['train_embeddings.npy', 'training_embeddings.npy'],
            'train_ids': ['train_ids.npy', 'training_ids.npy'],
            'test_embeddings': ['test_embeddings.npy', 'testing_embeddings.npy'],
            'test_ids': ['test_ids.npy', 'testing_ids.npy'],
        }
        
        # Search recursively
        for dataset_type, filenames in patterns.items():
            for filename in filenames:
                matches = list(path.rglob(filename))
                if matches:
                    datasets[dataset_type] = matches[0]
                    break
        
        return datasets
    
    def validate_datasets(self) -> Dict[str, bool]:
        """Validate found datasets."""
        if self.verbose:
            print("\n🔍 VALIDATING DATASETS")
            print("="*50)
        
        validation = {}
        
        for name, path in self.dataset_paths.items():
            try:
                is_valid = self._validate_file(path, name)
                validation[name] = is_valid
                
                if self.verbose:
                    status = "✓" if is_valid else "✗"
                    print(f"  {status} {name}: {path}")
                    
            except Exception as e:
                validation[name] = False
                if self.verbose:
                    print(f"  ✗ {name}: ERROR - {e}")
        
        self.validation_results = validation
        return validation
    
    def _validate_file(self, path: Path, dataset_type: str) -> bool:
        """Validate specific file types."""
        if not path.exists():
            return False
        
        if path.stat().st_size == 0:
            return False
        
        if dataset_type == 'go_ontology':
            return self._validate_go_obo(path)
        elif dataset_type in ['train_terms', 'ia_weights']:
            return self._validate_tsv(path)
        elif 'embeddings' in dataset_type or 'ids' in dataset_type:
            return self._validate_npy(path)
        
        return True
    
    def _validate_go_obo(self, path: Path) -> bool:
        """Validate GO ontology file."""
        try:
            with open(path, 'r') as f:
                first_lines = f.read(1000)
                return 'format-version:' in first_lines and '[Term]' in first_lines
        except:
            return False
    
    def _validate_tsv(self, path: Path) -> bool:
        """Validate TSV files."""
        try:
            df = pd.read_csv(path, sep='\t', nrows=5)
            return len(df.columns) >= 2 and len(df) > 0
        except:
            return False
    
    def _validate_npy(self, path: Path) -> bool:
        """Validate numpy files."""
        try:
            arr = np.load(path, mmap_mode='r')
            return arr.size > 0
        except:
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary."""
        summary = {
            'total_found': len(self.dataset_paths),
            'valid_count': sum(self.validation_results.values()) if self.validation_results else 0,
            'datasets': {}
        }
        
        for name, path in self.dataset_paths.items():
            is_valid = self.validation_results.get(name, False)
            file_size = path.stat().st_size if path.exists() else 0
            
            summary['datasets'][name] = {
                'path': str(path),
                'valid': is_valid,
                'size_mb': file_size / (1024 * 1024)
            }
        
        return summary

# =============================================================================
# Feature 2: Bayesian Hyperparameter Optimization
# =============================================================================

@dataclass
class HyperparameterSpace:
    """Define search space for a hyperparameter."""
    name: str
    min_val: float
    max_val: float
    dtype: type = float
    log_scale: bool = False

class BayesianOptimizer:
    """Bayesian hyperparameter optimization for CAFA-6 models."""
    
    def __init__(self, 
                 param_spaces: List[HyperparameterSpace],
                 objective_func: Callable,
                 n_initial: int = 5,
                 n_iterations: int = 15):
        self.param_spaces = param_spaces
        self.objective_func = objective_func
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.trial_history = []
        
        if not BAYESIAN_OPT_AVAILABLE:
            warnings.warn("Bayesian optimization unavailable - using random search")
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization loop."""
        print("\n🎯 BAYESIAN HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        best_score = -np.inf
        best_params = None
        
        for trial_num in range(self.n_initial + self.n_iterations):
            print(f"\nTrial {trial_num + 1}/{self.n_initial + self.n_iterations}")
            
            # Suggest parameters
            params = self._suggest_params(trial_num)
            print(f"  Params: {params}")
            
            # Evaluate objective
            start_time = time.time()
            try:
                score = self.objective_func(params)
                elapsed = time.time() - start_time
                
                print(f"  Score: {score:.4f} (took {elapsed:.1f}s)")
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  🏆 New best score!")
                
            except Exception as e:
                score = -np.inf
                elapsed = time.time() - start_time
                print(f"  ❌ Failed: {e}")
            
            # Record result
            self.trial_history.append({
                'trial': trial_num,
                'params': params,
                'score': score,
                'elapsed': elapsed
            })
        
        print(f"\n🎯 Optimization Complete!")
        print(f"   Best Score: {best_score:.4f}")
        print(f"   Best Params: {best_params}")
        
        return best_params
    
    def _suggest_params(self, trial_num: int) -> Dict[str, Any]:
        """Suggest next parameters."""
        if trial_num < self.n_initial or not BAYESIAN_OPT_AVAILABLE:
            return self._random_params()
        
        return self._bayesian_suggest()
    
    def _random_params(self) -> Dict[str, Any]:
        """Generate random parameters."""
        params = {}
        
        for space in self.param_spaces:
            if space.log_scale:
                val = np.exp(np.random.uniform(np.log(space.min_val), np.log(space.max_val)))
            else:
                val = np.random.uniform(space.min_val, space.max_val)
            
            if space.dtype == int:
                val = int(round(val))
            
            params[space.name] = val
        
        return params
    
    def _bayesian_suggest(self) -> Dict[str, Any]:
        """Use Gaussian Process to suggest parameters."""
        # Prepare training data
        X = []
        y = []
        
        for trial in self.trial_history:
            if trial['score'] != -np.inf:
                x = [trial['params'][s.name] for s in self.param_spaces]
                X.append(x)
                y.append(trial['score'])
        
        if len(X) < 2:
            return self._random_params()
        
        # Fit Gaussian Process
        X = np.array(X)
        y = np.array(y)
        
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6)
        gp.fit(X, y)
        
        # Sample candidates and compute acquisition
        candidates = []
        for _ in range(1000):
            candidate = []
            for space in self.param_spaces:
                if space.log_scale:
                    val = np.exp(np.random.uniform(np.log(space.min_val), np.log(space.max_val)))
                else:
                    val = np.random.uniform(space.min_val, space.max_val)
                candidate.append(val)
            candidates.append(candidate)
        
        candidates = np.array(candidates)
        mu, sigma = gp.predict(candidates, return_std=True)
        
        # UCB acquisition function
        kappa = 2.0
        acquisition = mu + kappa * sigma
        
        best_idx = np.argmax(acquisition)
        best_candidate = candidates[best_idx]
        
        # Convert back to parameter dict
        params = {}
        for i, space in enumerate(self.param_spaces):
            val = best_candidate[i]
            if space.dtype == int:
                val = int(round(val))
            params[space.name] = val
        
        return params

# CAFA-6 specific parameter spaces
CAFA6_PARAM_SPACES = [
    HyperparameterSpace("learning_rate", 1e-5, 1e-2, float, log_scale=True),
    HyperparameterSpace("batch_size", 16, 128, int, log_scale=False),
    HyperparameterSpace("hidden_size", 256, 2048, int, log_scale=False), 
    HyperparameterSpace("dropout", 0.1, 0.7, float),
    HyperparameterSpace("weight_decay", 1e-6, 1e-3, float, log_scale=True),
]

# =============================================================================
# Feature 3: Feature/Operation Importance Tracking
# =============================================================================

class FeatureImportanceTracker:
    """Track feature importance for CAFA-6 pipeline."""
    
    def __init__(self):
        self.feature_stats = defaultdict(lambda: {
            'used': 0,
            'success': 0,
            'failure': 0,
            'avg_time_ms': 0.0,
            'contexts': set(),
            'performance_scores': []
        })
    
    def record_feature(self, 
                      feature_name: str,
                      success: bool,
                      time_ms: float,
                      context: str = "default",
                      performance_score: float = None):
        """Record feature usage and outcome."""
        stats = self.feature_stats[feature_name]
        stats['used'] += 1
        
        if success:
            stats['success'] += 1
        else:
            stats['failure'] += 1
        
        # Update running average time
        n = stats['used']
        stats['avg_time_ms'] = ((n - 1) * stats['avg_time_ms'] + time_ms) / n
        stats['contexts'].add(context)
        
        if performance_score is not None:
            stats['performance_scores'].append(performance_score)
    
    def get_importance_scores(self) -> List[Tuple[str, float]]:
        """Calculate importance scores for all features."""
        importance = []
        
        for feature, stats in self.feature_stats.items():
            if stats['used'] == 0:
                continue
            
            success_rate = stats['success'] / stats['used']
            usage_score = np.log1p(stats['used'])
            efficiency = 1.0 / max(stats['avg_time_ms'], 0.001)
            versatility = len(stats['contexts'])
            
            # Performance score factor
            perf_factor = 1.0
            if stats['performance_scores']:
                perf_factor = np.mean(stats['performance_scores'])
            
            score = (success_rate * usage_score * 
                    np.log1p(efficiency) * np.log1p(versatility) * perf_factor)
            
            importance.append((feature, score))
        
        return sorted(importance, key=lambda x: x[1], reverse=True)
    
    def print_report(self, top_n: int = 15):
        """Print importance report."""
        top_features = self.get_importance_scores()[:top_n]
        
        print(f"\n🏆 TOP {len(top_features)} MOST IMPORTANT FEATURES")
        print("="*80)
        print(f"{'Rank':<5} {'Feature':<30} {'Score':<10} {'Success%':<10} {'Used':<8} {'AvgTime':<10}")
        print("-" * 80)
        
        for rank, (feature, score) in enumerate(top_features, 1):
            stats = self.feature_stats[feature]
            success_pct = 100 * stats['success'] / stats['used']
            print(f"{rank:<5} {feature:<30} {score:<10.2f} "
                  f"{success_pct:<10.1f} {stats['used']:<8} {stats['avg_time_ms']:<10.1f}ms")

# =============================================================================
# Feature 4: Multi-Model Ensemble System
# =============================================================================

class ModelType(Enum):
    """Types of models in CAFA-6 ensemble."""
    MLP_BASIC = "mlp_basic"
    MLP_HIERARCHICAL = "mlp_hierarchical"
    TRANSFORMER = "transformer"
    ENSEMBLE_BLEND = "ensemble_blend"
    HOMOLOGY_ENHANCED = "homology_enhanced"

@dataclass
class ModelResult:
    """Result from a single model."""
    model_type: ModelType
    prediction: Any
    confidence: float  # 0-1
    time_ms: float
    metadata: Dict = None

class EnsembleSystem:
    """Multi-model ensemble for CAFA-6."""
    
    def __init__(self, 
                 models: List[Tuple[ModelType, Callable]],
                 voting_strategy: str = "weighted_confidence"):
        self.models = models
        self.voting_strategy = voting_strategy
        self.model_stats = defaultdict(lambda: {
            'attempts': 0,
            'correct': 0,
            'avg_confidence': 0.0,
            'avg_time_ms': 0.0,
            'avg_performance': 0.0
        })
    
    def predict(self, X: Any, y_true: Any = None) -> ModelResult:
        """Get predictions from all models and ensemble them."""
        results = []
        
        # Get predictions from all models
        for model_type, model_func in self.models:
            try:
                start = time.time()
                prediction, confidence, metadata = model_func(X)
                elapsed_ms = (time.time() - start) * 1000
                
                result = ModelResult(model_type, prediction, 
                                   confidence, elapsed_ms, metadata)
                results.append(result)
                
                # Update statistics
                self._update_stats(model_type, result, y_true)
                
            except Exception as e:
                print(f"⚠️  Model {model_type.value} failed: {e}")
                continue
        
        if not results:
            raise RuntimeError("All models failed!")
        
        # Ensemble voting
        best_result = self._vote(results)
        return best_result
    
    def _vote(self, results: List[ModelResult]) -> ModelResult:
        """Apply voting strategy to select best prediction."""
        if self.voting_strategy == "weighted_confidence":
            return self._weighted_confidence_vote(results)
        elif self.voting_strategy == "average_blend":
            return self._average_blend(results)
        else:
            # Default: highest confidence
            return max(results, key=lambda r: r.confidence)
    
    def _weighted_confidence_vote(self, results: List[ModelResult]) -> ModelResult:
        """Vote based on confidence × historical accuracy."""
        best_result = None
        best_score = -1
        
        for result in results:
            stats = self.model_stats[result.model_type]
            
            # Historical accuracy
            accuracy = (stats['correct'] / stats['attempts'] 
                       if stats['attempts'] > 0 else 0.5)
            
            # Combined score
            score = result.confidence * 0.7 + accuracy * 0.3
            
            # Time penalty for slow models
            time_penalty = 1.0 / (1.0 + result.time_ms / 5000)
            score *= time_penalty
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def _average_blend(self, results: List[ModelResult]) -> ModelResult:
        """Blend predictions using weighted average."""
        if not results:
            raise ValueError("No results to blend")
        
        # Get weights based on historical performance
        weights = []
        predictions = []
        total_time = 0
        
        for result in results:
            stats = self.model_stats[result.model_type]
            weight = max(stats['avg_performance'], 0.1)  # Minimum weight
            weights.append(weight)
            predictions.append(result.prediction)
            total_time += result.time_ms
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Blend predictions
        if isinstance(predictions[0], np.ndarray):
            blended_pred = np.average(predictions, axis=0, weights=weights)
        else:
            # For scalar predictions
            blended_pred = np.average(predictions, weights=weights)
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in results])
        
        return ModelResult(
            ModelType.ENSEMBLE_BLEND,
            blended_pred,
            avg_confidence,
            total_time,
            {'method': 'weighted_average', 'n_models': len(results)}
        )
    
    def _update_stats(self, model_type: ModelType, result: ModelResult, y_true: Any):
        """Update model statistics."""
        stats = self.model_stats[model_type]
        stats['attempts'] += 1
        
        # Update time
        n = stats['attempts']
        stats['avg_time_ms'] = ((n - 1) * stats['avg_time_ms'] + result.time_ms) / n
        
        # Update confidence
        stats['avg_confidence'] = ((n - 1) * stats['avg_confidence'] + result.confidence) / n
        
        # If ground truth available, update correctness
        if y_true is not None:
            # This would need to be customized based on your metric
            # For now, we'll use a placeholder
            is_correct = True  # Implement your evaluation logic
            if is_correct:
                stats['correct'] += 1
    
    def print_stats(self):
        """Print ensemble statistics."""
        print(f"\n📊 ENSEMBLE MODEL STATISTICS")
        print("="*80)
        print(f"{'Model':<25} {'Attempts':<10} {'Accuracy':<10} {'AvgTime':<12} {'AvgConf':<10}")
        print("-" * 80)
        
        for model_type, stats in self.model_stats.items():
            if stats['attempts'] > 0:
                accuracy = 100 * stats['correct'] / stats['attempts']
                print(f"{model_type.value:<25} {stats['attempts']:<10} "
                      f"{accuracy:<10.1f}% {stats['avg_time_ms']:<12.1f}ms {stats['avg_confidence']:<10.3f}")

# =============================================================================
# Feature 5: Performance Monitoring Dashboard
# =============================================================================

class PerformanceMonitor:
    """Real-time performance monitoring for CAFA-6 training."""
    
    def __init__(self, total_tasks: int = 0, window_size: int = 20):
        self.total_tasks = total_tasks
        self.window_size = window_size
        
        # Overall metrics
        self.tasks_attempted = 0
        self.tasks_successful = 0
        self.start_time = time.time()
        
        # Rolling window (recent performance)
        self.recent_times = deque(maxlen=window_size)
        self.recent_successes = deque(maxlen=window_size)
        self.recent_scores = deque(maxlen=window_size)
        
        # Detailed log
        self.task_log = []
        
        # Breakdown by method/model
        self.method_breakdown = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'total_time': 0.0,
            'scores': []
        })
    
    def record_task(self,
                   task_id: str,
                   success: bool,
                   time_ms: float,
                   method: str = "default",
                   confidence: float = 0.0,
                   score: float = None,
                   metadata: Dict = None):
        """Record task result and update metrics."""
        self.tasks_attempted += 1
        if success:
            self.tasks_successful += 1
        
        # Update rolling window
        self.recent_times.append(time_ms)
        self.recent_successes.append(1 if success else 0)
        if score is not None:
            self.recent_scores.append(score)
        
        # Update method breakdown
        self.method_breakdown[method]['attempts'] += 1
        self.method_breakdown[method]['total_time'] += time_ms
        if success:
            self.method_breakdown[method]['successes'] += 1
        if score is not None:
            self.method_breakdown[method]['scores'].append(score)
        
        # Detailed log
        self.task_log.append({
            'task_id': task_id,
            'timestamp': datetime.now(),
            'success': success,
            'time_ms': time_ms,
            'method': method,
            'confidence': confidence,
            'score': score,
            'metadata': metadata or {}
        })
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Calculate current performance statistics."""
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'tasks_attempted': self.tasks_attempted,
            'tasks_successful': self.tasks_successful,
            'success_rate': (self.tasks_successful / self.tasks_attempted 
                            if self.tasks_attempted > 0 else 0),
            'elapsed_time_s': elapsed_time,
            'avg_time_ms': (np.mean(self.recent_times) 
                           if self.recent_times else 0),
            'recent_success_rate': (np.mean(self.recent_successes) 
                                   if self.recent_successes else 0),
            'throughput_per_min': ((self.tasks_attempted / elapsed_time) * 60 
                                  if elapsed_time > 0 else 0),
        }
        
        # Add recent score if available
        if self.recent_scores:
            stats['recent_avg_score'] = np.mean(self.recent_scores)
            stats['recent_best_score'] = np.max(self.recent_scores)
        
        # ETA calculation
        if self.total_tasks > 0:
            stats['progress_pct'] = 100 * self.tasks_attempted / self.total_tasks
            remaining = self.total_tasks - self.tasks_attempted
            if stats['throughput_per_min'] > 0:
                stats['eta_minutes'] = remaining / stats['throughput_per_min']
        
        return stats
    
    def print_dashboard(self, detailed: bool = True):
        """Print visual performance dashboard."""
        stats = self.get_current_stats()
        
        print("\n" + "="*70)
        print("📈 CAFA-6 PERFORMANCE DASHBOARD")
        print("="*70)
        
        # Main metrics
        print(f"\n⚡ Overall Performance:")
        print(f"   Tasks: {stats['tasks_attempted']} attempted, "
              f"{stats['tasks_successful']} successful")
        print(f"   Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"   Elapsed: {stats['elapsed_time_s']:.1f}s")
        
        # Progress (if total known)
        if self.total_tasks > 0:
            print(f"\n📊 Progress:")
            bar_length = 40
            progress = int(bar_length * stats['progress_pct'] / 100)
            bar = "█" * progress + "░" * (bar_length - progress)
            print(f"   [{bar}] {stats['progress_pct']:.1f}%")
            if 'eta_minutes' in stats:
                print(f"   ETA: {stats['eta_minutes']:.1f} minutes")
        
        # Recent performance
        print(f"\n🔥 Recent (last {self.window_size}):")
        print(f"   Avg Time: {stats['avg_time_ms']:.1f}ms")
        print(f"   Success Rate: {stats['recent_success_rate']*100:.1f}%")
        print(f"   Throughput: {stats['throughput_per_min']:.1f} tasks/min")
        
        if 'recent_avg_score' in stats:
            print(f"   Avg Score: {stats['recent_avg_score']:.4f}")
            print(f"   Best Score: {stats['recent_best_score']:.4f}")
        
        # Method breakdown
        if detailed and self.method_breakdown:
            print(f"\n🏆 Method Breakdown:")
            print(f"   {'Method':<20} {'Attempts':<10} {'Success':<10} {'Rate%':<8} {'AvgTime':<10}")
            print("   " + "-"*60)
            
            for method, data in sorted(self.method_breakdown.items(),
                                       key=lambda x: x[1]['successes'],
                                       reverse=True):
                attempts = data['attempts']
                successes = data['successes']
                rate = 100 * successes / attempts if attempts > 0 else 0
                avg_time = data['total_time'] / attempts if attempts > 0 else 0
                print(f"   {method:<20} {attempts:<10} {successes:<10} "
                      f"{rate:<8.1f} {avg_time:<10.1f}ms")
        
        print("="*70)
    
    def export_log(self, filepath: str):
        """Export detailed log to JSON."""
        # Convert timestamps to strings for JSON serialization
        log_data = []
        for entry in self.task_log:
            entry_copy = entry.copy()
            entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
            log_data.append(entry_copy)
        
        with open(filepath, 'w') as f:
            json.dump({
                'stats': self.get_current_stats(),
                'method_breakdown': dict(self.method_breakdown),
                'task_log': log_data
            }, f, indent=2)
        
        print(f"📄 Performance log exported to: {filepath}")

# =============================================================================
# Feature 6: Advanced Submission Generation
# =============================================================================

class SubmissionGenerator:
    """Generate CAFA-6 competition submissions with validation."""
    
    def __init__(self, output_dir: str = "./submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.predictions = {}
        self.metadata = {}
        self.cafa6_format = True  # Enable CAFA-6 specific formatting
    
    def add_prediction(self,
                      protein_id: str,
                      go_predictions: List[Tuple[str, float]],
                      confidence: float = 1.0,
                      method: str = "default",
                      metadata: Dict = None):
        """
        Add CAFA-6 prediction for a protein.
        
        Args:
            protein_id: Protein identifier
            go_predictions: List of (GO_term, score) tuples
            confidence: Overall confidence in prediction
            method: Method used for prediction
            metadata: Additional metadata
        """
        # Ensure max 1500 predictions per protein (CAFA-6 rule)
        if len(go_predictions) > 1500:
            go_predictions = sorted(go_predictions, key=lambda x: x[1], reverse=True)[:1500]
        
        self.predictions[protein_id] = {
            'go_predictions': go_predictions,
            'confidence': confidence,
            'method': method,
            'metadata': metadata or {}
        }
    
    def generate_cafa6_tsv(self, filename: str = "submission.tsv") -> Path:
        """
        Generate CAFA-6 competition TSV submission.
        
        Format:
        T101900012345    GO:0003674    0.95
        T101900012345    GO:0008150    0.87
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for protein_id, data in sorted(self.predictions.items()):
                for go_term, score in data['go_predictions']:
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")
        
        print(f"✅ CAFA-6 TSV submission saved: {filepath}")
        return filepath
    
    def generate_detailed_report(self, filename: str = "submission_report.json") -> Path:
        """Generate detailed report with all metadata."""
        report = {
            'submission_info': {
                'timestamp': datetime.now().isoformat(),
                'total_proteins': len(self.predictions),
                'total_predictions': sum(len(d['go_predictions']) for d in self.predictions.values()),
                'avg_confidence': np.mean([d['confidence'] for d in self.predictions.values()]),
                'avg_predictions_per_protein': np.mean([len(d['go_predictions']) for d in self.predictions.values()]),
            },
            'predictions': {}
        }
        
        for protein_id, data in self.predictions.items():
            report['predictions'][protein_id] = {
                'go_predictions': data['go_predictions'],
                'confidence': data['confidence'],
                'method': data['method'],
                'metadata': data['metadata']
            }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Detailed report saved: {filepath}")
        return filepath
    
    def validate_submission(self) -> Dict[str, Any]:
        """Validate CAFA-6 submission format and quality."""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check basics
        if not self.predictions:
            validation['valid'] = False
            validation['errors'].append("No predictions in submission")
            return validation
        
        total_predictions = 0
        protein_count = 0
        go_terms = set()
        score_issues = 0
        
        # Check each protein prediction
        for protein_id, data in self.predictions.items():
            protein_count += 1
            go_predictions = data['go_predictions']
            
            # Check prediction count per protein
            if len(go_predictions) > 1500:
                validation['errors'].append(
                    f"Protein {protein_id}: Too many predictions ({len(go_predictions)} > 1500)"
                )
                validation['valid'] = False
            
            if len(go_predictions) == 0:
                validation['warnings'].append(
                    f"Protein {protein_id}: No predictions"
                )
            
            # Check GO terms and scores
            for go_term, score in go_predictions:
                total_predictions += 1
                go_terms.add(go_term)
                
                # Validate GO term format
                if not go_term.startswith('GO:') or len(go_term) != 10:
                    validation['warnings'].append(
                        f"Protein {protein_id}: Invalid GO term format '{go_term}'"
                    )
                
                # Validate score range
                if not 0 <= score <= 1:
                    validation['warnings'].append(
                        f"Protein {protein_id}: Score {score} out of range [0,1] for {go_term}"
                    )
                    score_issues += 1
        
        # Statistics
        validation['stats'] = {
            'total_proteins': protein_count,
            'total_predictions': total_predictions,
            'unique_go_terms': len(go_terms),
            'avg_predictions_per_protein': total_predictions / protein_count if protein_count > 0 else 0,
            'score_issues': score_issues,
            'avg_confidence': np.mean([d['confidence'] for d in self.predictions.values()]),
            'method_distribution': {}
        }
        
        # Method distribution
        for data in self.predictions.values():
            method = data['method']
            validation['stats']['method_distribution'][method] = \
                validation['stats']['method_distribution'].get(method, 0) + 1
        
        return validation
    
    def print_validation_report(self):
        """Print human-readable validation report."""
        report = self.validate_submission()
        
        print("\n" + "="*70)
        print("✔️  CAFA-6 SUBMISSION VALIDATION REPORT")
        print("="*70)
        
        if report['valid']:
            print("\n✅ Submission is VALID for CAFA-6 competition")
        else:
            print("\n❌ Submission has ERRORS - FIX BEFORE SUBMITTING")
        
        # Errors
        if report['errors']:
            print(f"\n🚨 Errors ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"   - {error}")
        
        # Warnings
        if report['warnings']:
            print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
            for warning in report['warnings'][:10]:
                print(f"   - {warning}")
            if len(report['warnings']) > 10:
                print(f"   ... and {len(report['warnings']) - 10} more warnings")
        
        # Stats
        stats = report['stats']
        print(f"\n📊 Submission Statistics:")
        print(f"   Proteins: {stats['total_proteins']}")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Unique GO Terms: {stats['unique_go_terms']}")
        print(f"   Avg Predictions/Protein: {stats['avg_predictions_per_protein']:.1f}")
        print(f"   Avg Confidence: {stats['avg_confidence']:.3f}")
        print(f"   Score Issues: {stats['score_issues']}")
        
        if stats['method_distribution']:
            print(f"\n🏆 Method Distribution:")
            for method, count in sorted(stats['method_distribution'].items(),
                                        key=lambda x: x[1], reverse=True):
                pct = 100 * count / stats['total_proteins']
                print(f"   {method:<20} {count:>5} proteins ({pct:>5.1f}%)")
        
        print("="*70)
    
    def generate_all_formats(self, base_name: str = "cafa6_submission"):
        """Generate all submission formats."""
        files_created = []
        
        # CAFA-6 TSV (main submission)
        tsv_file = self.generate_cafa6_tsv(f"{base_name}.tsv")
        files_created.append(tsv_file)
        
        # Detailed report
        report_file = self.generate_detailed_report(f"{base_name}_report.json")
        files_created.append(report_file)
        
        # Validation
        self.print_validation_report()
        
        print(f"\n📁 All files generated in: {self.output_dir}")
        return files_created

# =============================================================================
# Integration Helper Class
# =============================================================================

class CAFA6ProfessionalPipeline:
    """Complete CAFA-6 pipeline with all professional features."""
    
    def __init__(self):
        self.detector = DatasetDetector(verbose=True)
        self.feature_tracker = FeatureImportanceTracker()
        self.monitor = PerformanceMonitor()
        self.submission = SubmissionGenerator()
        self.ensemble = None  # Initialize when models are ready
        self.optimizer = None  # Initialize when needed
    
    def setup_environment(self):
        """Setup and validate environment."""
        print("\n🚀 CAFA-6 PROFESSIONAL PIPELINE SETUP")
        print("="*60)
        
        # Detect datasets
        datasets = self.detector.scan_directories()
        valid_datasets = self.detector.validate_datasets()
        
        summary = self.detector.get_summary()
        print(f"\n📊 Dataset Summary:")
        print(f"   Found: {summary['total_found']} datasets")
        print(f"   Valid: {summary['valid_count']} datasets")
        
        return datasets, valid_datasets
    
    def setup_hyperparameter_optimization(self, objective_func: Callable):
        """Setup Bayesian optimization."""
        self.optimizer = BayesianOptimizer(
            param_spaces=CAFA6_PARAM_SPACES,
            objective_func=objective_func,
            n_initial=5,
            n_iterations=15
        )
        return self.optimizer
    
    def setup_ensemble(self, models: List[Tuple[ModelType, Callable]]):
        """Setup ensemble system."""
        self.ensemble = EnsembleSystem(
            models=models,
            voting_strategy="weighted_confidence"
        )
        return self.ensemble
    
    def track_feature_performance(self, feature_name: str, success: bool, 
                                time_ms: float, context: str = "training",
                                performance_score: float = None):
        """Track feature performance."""
        self.feature_tracker.record_feature(
            feature_name, success, time_ms, context, performance_score
        )
    
    def monitor_training(self, task_id: str, success: bool, time_ms: float,
                        method: str = "default", confidence: float = 0.0,
                        score: float = None):
        """Monitor training progress."""
        self.monitor.record_task(
            task_id, success, time_ms, method, confidence, score
        )
    
    def generate_final_reports(self):
        """Generate all final reports."""
        print("\n📊 GENERATING FINAL REPORTS")
        print("="*60)
        
        # Feature importance report
        self.feature_tracker.print_report()
        
        # Performance dashboard
        self.monitor.print_dashboard(detailed=True)
        
        # Ensemble statistics
        if self.ensemble:
            self.ensemble.print_stats()
        
        # Export logs
        self.monitor.export_log("/kaggle/working/performance_log.json")
        
        # Validate submission
        self.submission.print_validation_report()
        
        # Generate submission files
        files = self.submission.generate_all_formats("cafa6_professional")
        
        return files

# =============================================================================
# Example Usage Functions
# =============================================================================

def create_cafa6_objective_function(train_func: Callable, val_func: Callable):
    """Create objective function for hyperparameter optimization."""
    
    def objective(params: Dict[str, Any]) -> float:
        """
        Objective function for CAFA-6 hyperparameter optimization.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Validation score (higher is better)
        """
        try:
            # Train model with given parameters
            model = train_func(params)
            
            # Evaluate on validation set
            score = val_func(model)
            
            return score
        
        except Exception as e:
            print(f"Objective function failed: {e}")
            return -np.inf
    
    return objective

def example_model_predictor(model, model_type: ModelType):
    """Create model predictor function for ensemble."""
    
    def predictor(X):
        """
        Model prediction function.
        
        Returns:
            tuple: (prediction, confidence, metadata)
        """
        try:
            prediction = model.predict(X)
            
            # Estimate confidence (customize based on your model)
            if hasattr(model, 'predict_proba'):
                confidence = np.mean(np.max(model.predict_proba(X), axis=1))
            else:
                confidence = 0.8  # Default confidence
            
            metadata = {'model_type': model_type.value}
            
            return prediction, confidence, metadata
            
        except Exception as e:
            raise RuntimeError(f"Model {model_type.value} prediction failed: {e}")
    
    return predictor

# =============================================================================
# Export all classes and functions
# =============================================================================

__all__ = [
    'DatasetDetector',
    'BayesianOptimizer', 
    'HyperparameterSpace',
    'CAFA6_PARAM_SPACES',
    'FeatureImportanceTracker',
    'EnsembleSystem',
    'ModelType',
    'ModelResult', 
    'PerformanceMonitor',
    'SubmissionGenerator',
    'CAFA6ProfessionalPipeline',
    'create_cafa6_objective_function',
    'example_model_predictor'
]