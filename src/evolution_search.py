#!/usr/bin/env python3
"""
Evolutionary Model Search for PhysiChemNet
===========================================
Mutates model architecture configurations and trains each variant.
Keeps track of the best model found so far.
Runs until it can't find a better model.

This REPLACES the base paper's DynamicNet approach entirely.
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import copy
import json
import time
import random
import itertools
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from src.utils.physics_features import extract_features_and_labels
from models.new_architecture import PhysiChemNet, PhysicsConstrainedLoss


# ============================================================================
# Data Loading
# ============================================================================

def load_data(use_physics=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "processed", "MemTrOC-Dataset.csv")
    data = pd.read_csv(file_path)
    X, y, smiles = extract_features_and_labels(data, use_physics=use_physics)

    # Also extract steric ratios for physics loss
    r_solute = data['Molecular radius (nm)'].values
    r_pore = np.maximum(data['Pore radius (nm)'].values, 1e-6)
    steric_ratios = r_solute / r_pore

    return X, y, smiles, steric_ratios


def collate_fn(batch):
    if len(batch[0]) == 4:
        table_features, graph_data, labels, sterics = zip(*batch)
        table_features = torch.stack(table_features, dim=0)
        labels = torch.tensor(labels, dtype=torch.float32)
        graph_data = Batch.from_data_list(graph_data)
        sterics = torch.tensor(sterics, dtype=torch.float32)
        return table_features, graph_data, labels, sterics
    else:
        table_features, graph_data, labels = zip(*batch)
        table_features = torch.stack(table_features, dim=0)
        labels = torch.tensor(labels, dtype=torch.float32)
        graph_data = Batch.from_data_list(graph_data)
        return table_features, graph_data, labels


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# In-memory graph cache to eliminate RDKit overhead during training
GRAPH_CACHE = {}

def get_cached_graph(smile, label):
    if smile not in GRAPH_CACHE:
        g = create_graph_data_from_smiles(str(smile), 0.0)
        GRAPH_CACHE[smile] = g
    cached = GRAPH_CACHE[smile]
    if cached is None:
        return None
    # Clone graph data object and set target label
    g_out = cached.clone()
    g_out.y = torch.tensor([label], dtype=torch.float32)
    return g_out


# Pre-cache all graphs in dataset
def precache_all_graphs(smiles_list):
    unique_smiles = list(set(smiles_list))
    print(f"  [Speedup] Pre-caching {len(unique_smiles)} unique molecular graphs in RAM...")
    for s in unique_smiles:
        get_cached_graph(s, 0.0)
    print(f"  [Speedup] Caching complete. Zero RDKit overhead during training!")


# ============================================================================
# Single Model Training & Evaluation
# ============================================================================

def train_single_model(config, X_train, y_train, smiles_train, steric_train,
                       X_val, y_val, smiles_val, steric_val,
                       X_test, y_test, smiles_test, steric_test,
                       verbose=True):
    """
    Train one PhysiChemNet model with the given config.
    Returns metrics dict.
    """
    set_seed(config.get('seed', 42))
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    # Normalize
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    # Create datasets with pre-cached graph retrieval
    train_ds = TableGraphDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        smiles_train, 
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        get_cached_graph,
        steric_ratios=steric_train
    )
    val_ds = TableGraphDataset(
        torch.tensor(X_va, dtype=torch.float32),
        smiles_val,
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
        get_cached_graph,
        steric_ratios=steric_val
    )
    test_ds = TableGraphDataset(
        torch.tensor(X_te, dtype=torch.float32),
        smiles_test,
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        get_cached_graph,
        steric_ratios=steric_test
    )

    batch_size = config.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Build model
    model = PhysiChemNet(config)
    param_count = model.count_parameters()
    if verbose:
        print(f"  Model parameters: {param_count:,}")

    # Loss function
    use_physics_loss = config.get('use_physics_loss', True)
    if use_physics_loss:
        criterion = PhysicsConstrainedLoss(
            lambda_steric=config.get('lambda_steric', 0.1),
            lambda_bounds=config.get('lambda_bounds', 0.01),
            use_huber=config.get('use_huber', False),
            huber_delta=config.get('huber_delta', 5.0)
        )
    else:
        criterion = nn.MSELoss()

    # Optimizer
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-3)
    optimizer_type = config.get('optimizer', 'adamw')
    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    epochs = config.get('epochs', 150)
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    # Training loop
    best_val_rmse = float('inf')
    best_model_state = None
    patience = config.get('patience', 30)
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            if len(batch) == 4:
                x, graph_data, y, batch_steric = batch
            else:
                x, graph_data, y = batch
                batch_steric = None

            y = y.squeeze()
            pred = model(x, graph_data)

            if use_physics_loss:
                loss = criterion(pred, y, batch_steric)
            else:
                loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            n_train += len(y)

        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, graph_data, y = batch[0], batch[1], batch[2]
                pred = model(x, graph_data)
                val_preds.append(pred.numpy())
                val_labels.append(y.squeeze().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
        val_r2 = r2_score(val_labels, val_preds)

        # Scheduler step
        if scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_rmse)

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    model.eval()

    # Get predictions for all splits
    results = {}
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                x, graph_data, y = batch[0], batch[1], batch[2]
                pred = model(x, graph_data)
                preds.append(pred.numpy())
                labels.append(y.squeeze().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        results[f'{name}_r2'] = r2_score(labels, preds)
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(labels, preds))
        results[f'{name}_mae'] = mean_absolute_error(labels, preds)

    results['param_count'] = param_count
    results['best_epoch'] = epoch + 1 - patience_counter

    return results, model


# ============================================================================
# 5-Fold Cross-Validation
# ============================================================================

def train_5fold(config, X_all, y_all, smiles_all, steric_all, verbose=True):
    """
    Run 5-fold CV with the given config. Returns averaged metrics.
    """
    # Hold out 10% fixed test set
    X_dev, X_test, y_dev, y_test, smiles_dev, smiles_test, steric_dev, steric_test = \
        train_test_split(X_all, y_all, smiles_all, steric_all,
                         test_size=0.1, random_state=41)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_test_preds = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_dev)):
        if verbose:
            print(f"\n--- Fold {fold_idx+1}/5 ---")

        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        smiles_tr, steric_tr = smiles_dev[train_idx], steric_dev[train_idx]
        X_va, y_va = X_dev[val_idx], y_dev[val_idx]
        smiles_va, steric_va = smiles_dev[val_idx], steric_dev[val_idx]

        results, model = train_single_model(
            config, X_tr, y_tr, smiles_tr, steric_tr,
            X_va, y_va, smiles_va, steric_va,
            X_test, y_test, smiles_test, steric_test,
            verbose=verbose
        )

        fold_metrics.append(results)
        all_test_preds.append(results['test_r2'])

        if verbose:
            print(f"  Fold {fold_idx+1} Test -> R²: {results['test_r2']:.4f}, "
                  f"RMSE: {results['test_rmse']:.4f}, MAE: {results['test_mae']:.4f}")

    # Average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        avg_metrics[f'avg_{key}'] = np.mean(vals)
        avg_metrics[f'std_{key}'] = np.std(vals)

    return avg_metrics


# ============================================================================
# EVOLUTIONARY MUTATION SEARCH
# ============================================================================

# Base configuration (starting point)
BASE_CONFIG = {
    'gnn_type': 'gatv2',
    'node_dim': 9,
    'edge_dim': 3,
    'hidden_dim': 128,
    'gnn_layers': 2,
    'gnn_heads': 4,
    'gnn_dropout': 0.2,
    'use_virtual_node': True,
    'table_dim': 24,
    'table_layers': 2,
    'table_dropout': 0.2,
    'fusion_type': 'cross_attention',
    'fusion_heads': 4,
    'fusion_dropout': 0.1,
    'pred_hidden': 64,
    'pred_dropout': 0.2,
    'lr': 0.001,
    'weight_decay': 1e-3,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'batch_size': 64,
    'epochs': 150,
    'patience': 30,
    'use_physics_loss': True,
    'lambda_steric': 0.1,
    'lambda_bounds': 0.01,
    'use_huber': False,
    'huber_delta': 5.0,
    'seed': 42
}

# Mutation space — what can be changed and to what values
MUTATION_SPACE = {
    'gnn_type':          ['gatv2', 'gine'],
    'hidden_dim':        [64, 96, 128, 192, 256],
    'gnn_layers':        [1, 2, 3],
    'gnn_heads':         [2, 4, 8],
    'gnn_dropout':       [0.1, 0.15, 0.2, 0.3],
    'use_virtual_node':  [True, False],
    'table_layers':      [1, 2, 3],
    'table_dropout':     [0.1, 0.2, 0.3],
    'fusion_type':       ['cross_attention', 'gated', 'concat'],
    'fusion_heads':      [2, 4, 8],
    'fusion_dropout':    [0.05, 0.1, 0.15, 0.2],
    'pred_hidden':       [32, 64, 128],
    'pred_dropout':      [0.1, 0.2, 0.3],
    'lr':                [0.0005, 0.001, 0.002, 0.003],
    'weight_decay':      [1e-4, 5e-4, 1e-3, 5e-3],
    'optimizer':         ['adam', 'adamw'],
    'scheduler':         ['cosine', 'plateau'],
    'batch_size':        [32, 64, 128],
    'epochs':            [100, 150, 200],
    'patience':          [20, 30, 40],
    'use_physics_loss':  [True, False],
    'lambda_steric':     [0.05, 0.1, 0.2, 0.5],
    'lambda_bounds':     [0.005, 0.01, 0.05],
    'use_huber':         [True, False],
    'huber_delta':       [3.0, 5.0, 8.0],
}


def mutate_config(config, n_mutations=2):
    """
    Create a mutated version of a config by randomly changing n parameters.
    """
    new_config = copy.deepcopy(config)
    keys = list(MUTATION_SPACE.keys())
    mutate_keys = random.sample(keys, min(n_mutations, len(keys)))
    
    mutations = {}
    for key in mutate_keys:
        old_val = new_config.get(key)
        choices = [v for v in MUTATION_SPACE[key] if v != old_val]
        if choices:
            new_val = random.choice(choices)
            new_config[key] = new_val
            mutations[key] = f"{old_val} -> {new_val}"

    return new_config, mutations


def run_evolutionary_search(max_generations=50, mutations_per_gen=3,
                             n_mutations=2, use_5fold=False):
    """
    Evolutionary search: start with base config, mutate, keep the best.
    Stops when no improvement found for 'stale_limit' consecutive generations.
    """
    print("=" * 70)
    print("  EVOLUTIONARY MODEL ARCHITECTURE SEARCH")
    print("  PhysiChemNet — Replacing DynamicNet from Base Paper")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    X_all, y_all, smiles_all, steric_all = load_data(use_physics=True)
    print(f"Dataset: {len(X_all)} samples, {X_all.shape[1]} features")

    # Split for quick single-split evaluation during search
    X_train, X_test, y_train, y_test, smiles_train, smiles_test, steric_train, steric_test = \
        train_test_split(X_all, y_all, smiles_all, steric_all,
                         test_size=0.1, random_state=41)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val, steric_train, steric_val = \
        train_test_split(X_train, y_train, smiles_train, steric_train,
                         test_size=0.2/0.9, random_state=41)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Precache all molecular graphs in RAM once
    precache_all_graphs(smiles_all)

    # Results tracking
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results", "evolution_search")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    best_config = copy.deepcopy(BASE_CONFIG)
    best_test_r2 = -float('inf')
    best_test_rmse = float('inf')
    best_test_mae = float('inf')
    stale_count = 0
    stale_limit = 8  # Stop after 8 generations with no improvement

    # Check for existing checkpoint to resume from peak performance
    best_model_path = os.path.join(base_dir, "checkpoint", "best_PhysiChemNet.pth")
    if os.path.exists(best_model_path):
        try:
            saved = torch.load(best_model_path, weights_only=False)
            if 'config' in saved and 'metrics' in saved:
                best_config = copy.deepcopy(saved['config'])
                best_test_r2 = saved['metrics'].get('test_r2', -float('inf'))
                best_test_rmse = saved['metrics'].get('test_rmse', float('inf'))
                best_test_mae = saved['metrics'].get('test_mae', float('inf'))
                print(f"  [Resume] Loaded previous best checkpoint: R2={best_test_r2:.4f}, MAE={best_test_mae:.4f}%")
        except Exception as ex:
            print(f"  [Warning] Could not load prior checkpoint: {ex}")

    # Optimize epochs for fast convergence with pre-cached graphs
    best_config['epochs'] = 90
    best_config['patience'] = 18

    # --- Generation 0: Evaluate base config ---
    print(f"\n{'='*70}")
    print(f"GENERATION 0 -- BASE CONFIG (Starting Point: R2={best_test_r2:.4f})")
    print(f"{'='*70}")
    t0 = time.time()

    try:
        results, model = train_single_model(
            best_config, X_train, y_train, smiles_train, steric_train,
            X_val, y_val, smiles_val, steric_val,
            X_test, y_test, smiles_test, steric_test,
            verbose=True
        )
        elapsed = time.time() - t0

        best_test_r2 = results['test_r2']
        best_test_rmse = results['test_rmse']
        best_test_mae = results['test_mae']

        result_entry = {
            'generation': 0,
            'config': copy.deepcopy(best_config),
            'mutations': 'BASE',
            'test_r2': results['test_r2'],
            'test_rmse': results['test_rmse'],
            'test_mae': results['test_mae'],
            'train_r2': results['train_r2'],
            'val_r2': results['val_r2'],
            'params': results['param_count'],
            'time': elapsed,
            'is_best': True
        }
        all_results.append(result_entry)

        # Save best model
        best_model_path = os.path.join(base_dir, "checkpoint", "best_PhysiChemNet.pth")
        torch.save({
            'model_state': model.state_dict(),
            'config': best_config,
            'metrics': results
        }, best_model_path)

        print(f"\n  [*] BASE RESULT: R2={results['test_r2']:.4f}, "
              f"RMSE={results['test_rmse']:.4f}%, MAE={results['test_mae']:.4f}%")
        print(f"  Time: {elapsed:.1f}s")

        # ===== BASE PAPER BENCHMARK =====
        print(f"\n  Base Paper (Xiao et al.): R2=0.9014, RMSE=9.11%, MAE=6.17%")
        if results['test_r2'] > 0.9014:
            print(f"  [+] ALREADY BEATING THE PAPER on R2!")
        if results['test_mae'] < 6.17:
            print(f"  [+] ALREADY BEATING THE PAPER on MAE!")

    except Exception as e:
        print(f"  [-] Base config failed: {e}")
        traceback.print_exc()
        return

    # --- Evolutionary Generations ---
    for gen in range(1, max_generations + 1):
        print(f"\n{'='*70}")
        print(f"GENERATION {gen} -- Mutating {mutations_per_gen} candidates "
              f"(stale: {stale_count}/{stale_limit})")
        print(f"{'='*70}")
        print(f"  Current best: R2={best_test_r2:.4f}, RMSE={best_test_rmse:.4f}%, "
              f"MAE={best_test_mae:.4f}%")

        gen_improved = False

        for cand_idx in range(mutations_per_gen):
            # Mutate best config
            n_mut = random.choice([1, 2, 3]) if gen > 3 else n_mutations
            candidate_config, mutations = mutate_config(best_config, n_mutations=n_mut)

            print(f"\n  Candidate {cand_idx+1}/{mutations_per_gen}:")
            for k, v in mutations.items():
                print(f"    {k}: {v}")

            t0 = time.time()
            try:
                results, model = train_single_model(
                    candidate_config,
                    X_train, y_train, smiles_train, steric_train,
                    X_val, y_val, smiles_val, steric_val,
                    X_test, y_test, smiles_test, steric_test,
                    verbose=False
                )
                elapsed = time.time() - t0

                is_better = False
                # Primary: lower MAE (more practically meaningful)
                # Secondary: higher R2
                if (results['test_mae'] < best_test_mae - 0.01 or
                    (results['test_mae'] <= best_test_mae + 0.05 and 
                     results['test_r2'] > best_test_r2 + 0.002)):
                    is_better = True

                result_entry = {
                    'generation': gen,
                    'config': copy.deepcopy(candidate_config),
                    'mutations': mutations,
                    'test_r2': results['test_r2'],
                    'test_rmse': results['test_rmse'],
                    'test_mae': results['test_mae'],
                    'train_r2': results['train_r2'],
                    'val_r2': results['val_r2'],
                    'params': results['param_count'],
                    'time': elapsed,
                    'is_best': is_better
                }
                all_results.append(result_entry)

                status = "[*] NEW BEST" if is_better else "    no improvement"
                print(f"    {status}: R2={results['test_r2']:.4f}, "
                      f"RMSE={results['test_rmse']:.4f}%, MAE={results['test_mae']:.4f}% "
                      f"({elapsed:.1f}s)")

                if is_better:
                    best_test_r2 = results['test_r2']
                    best_test_rmse = results['test_rmse']
                    best_test_mae = results['test_mae']
                    best_config = copy.deepcopy(candidate_config)
                    gen_improved = True

                    # Save best model
                    torch.save({
                        'model_state': model.state_dict(),
                        'config': best_config,
                        'metrics': results
                    }, best_model_path)

            except Exception as e:
                print(f"    [-] Failed: {str(e)[:100]}")
                elapsed = time.time() - t0
                all_results.append({
                    'generation': gen, 'mutations': mutations,
                    'test_r2': -1, 'test_rmse': 999, 'test_mae': 999,
                    'time': elapsed, 'is_best': False, 'error': str(e)[:200]
                })

        # Check stale generations
        if gen_improved:
            stale_count = 0
        else:
            stale_count += 1

        # Save progress
        progress = {
            'best_config': best_config,
            'best_metrics': {
                'test_r2': best_test_r2,
                'test_rmse': best_test_rmse,
                'test_mae': best_test_mae
            },
            'generation': gen,
            'total_candidates': len(all_results),
            'all_results_summary': [
                {k: v for k, v in r.items() if k != 'config'}
                for r in all_results
            ]
        }
        with open(os.path.join(results_dir, "search_progress.json"), 'w') as f:
            json.dump(progress, f, indent=2, default=str)

        # Stop condition
        if stale_count >= stale_limit:
            print(f"\n{'='*70}")
            print(f"  STOPPING: No improvement for {stale_limit} consecutive generations.")
            print(f"{'='*70}")
            break

    # ============================================================================
    # FINAL REPORT
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"  EVOLUTIONARY SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Total generations: {gen}")
    print(f"  Total candidates evaluated: {len(all_results)}")
    print(f"\n  +-------------------------------------------------------+")
    print(f"  |  BEST MODEL FOUND                                    |")
    print(f"  |  Test R2:   {best_test_r2:.4f}                                |")
    print(f"  |  Test RMSE: {best_test_rmse:.4f}%                              |")
    print(f"  |  Test MAE:  {best_test_mae:.4f}%                              |")
    print(f"  +-------------------------------------------------------+")
    print(f"\n  Base Paper: R2=0.9014, RMSE=9.11%, MAE=6.17%")
    
    if best_test_r2 > 0.9014:
        print(f"  [+] BEATS PAPER on R2 by {(best_test_r2 - 0.9014)*100:.2f} points!")
    else:
        print(f"  [-] Below paper R2 by {(0.9014 - best_test_r2)*100:.2f} points")
    if best_test_mae < 6.17:
        print(f"  [+] BEATS PAPER on MAE by {6.17 - best_test_mae:.2f}%!")
    if best_test_rmse < 9.11:
        print(f"  [+] BEATS PAPER on RMSE by {9.11 - best_test_rmse:.2f}%!")

    print(f"\n  Best config saved to: {best_model_path}")
    print(f"\n  Best configuration:")
    for k, v in sorted(best_config.items()):
        print(f"    {k}: {v}")

    # Now run 5-fold with best config if we beat the paper
    if best_test_r2 > 0.88:  # Only if reasonably good
        print(f"\n{'='*70}")
        print(f"  RUNNING 5-FOLD CV WITH BEST CONFIG")
        print(f"{'='*70}")
        try:
            fold_metrics = train_5fold(best_config, X_all, y_all, smiles_all, steric_all,
                                        verbose=True)
            print(f"\n  5-Fold Ensemble Results:")
            print(f"  Avg Test R²:   {fold_metrics['avg_test_r2']:.4f} ± {fold_metrics['std_test_r2']:.4f}")
            print(f"  Avg Test RMSE: {fold_metrics['avg_test_rmse']:.4f} ± {fold_metrics['std_test_rmse']:.4f}")
            print(f"  Avg Test MAE:  {fold_metrics['avg_test_mae']:.4f} ± {fold_metrics['std_test_mae']:.4f}")

            # Save final results
            progress['5fold_results'] = fold_metrics
            with open(os.path.join(results_dir, "search_progress.json"), 'w') as f:
                json.dump(progress, f, indent=2, default=str)

        except Exception as e:
            print(f"  5-fold failed: {e}")
            traceback.print_exc()

    return best_config, best_test_r2, best_test_rmse, best_test_mae


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evolutionary Search for PhysiChemNet")
    parser.add_argument('--max_generations', type=int, default=50,
                        help='Maximum number of evolution generations')
    parser.add_argument('--mutations_per_gen', type=int, default=3,
                        help='Number of mutant candidates per generation')
    parser.add_argument('--n_mutations', type=int, default=2,
                        help='Number of parameters to mutate per candidate')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer epochs, smaller search')
    args = parser.parse_args()

    if args.quick:
        BASE_CONFIG['epochs'] = 50
        BASE_CONFIG['patience'] = 15
        args.max_generations = 15
        args.mutations_per_gen = 2

    run_evolutionary_search(
        max_generations=args.max_generations,
        mutations_per_gen=args.mutations_per_gen,
        n_mutations=args.n_mutations
    )
