"""Experiment C: Deep Learning Time-Series Models for WESAD Stress Detection.
Evaluates 1D-CNN, TCN, and BiGRU on raw wrist sensor time-series (6 channels, 32Hz, 60s windows).
Uses 15-fold LOSO cross-validation on NVIDIA RTX 4060 GPU with CUDA.
Strict zero-leakage: normalizes using train fold statistics only.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.deep_timeseries import CNN1DStressModel, TCNStressModel, GRUStressModel

OUT_DIR = ROOT / 'results' / 'phase3' / 'deep_learning'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_b, y_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            out = model(x_b)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
    infer_time_ms = ((time.perf_counter() - t0) / len(all_probs)) * 1000.0 if all_probs else 0.0
    return np.array(all_probs), np.array(all_preds), infer_time_ms

def main():
    npz_path = ROOT / 'data' / 'processed' / 'wesad_dl_tensors.npz'
    if not npz_path.exists():
        print(f"Error: {npz_path} not found.")
        sys.exit(1)
        
    data = np.load(npz_path)
    X = data['X_wrist'] # shape (2031, 6, 1920)
    y = data['y']       # shape (2031,)
    subjects = data['subjects']
    unique_subs = sorted(list(set(subjects)))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    model_factories = {
        '1d_cnn': lambda: CNN1DStressModel(in_channels=6, num_classes=2, dropout=0.3),
        'tcn': lambda: TCNStressModel(in_channels=6, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2),
        'gru': lambda: GRUStressModel(in_channels=6, hidden_size=64, num_layers=2, dropout=0.2)
    }
    
    EPOCHS = 15
    BATCH_SIZE = 64
    LR = 1e-3
    
    summary_records = []
    per_sub_records = []
    
    print("================== EXPERIMENT C: DEEP LEARNING TIME-SERIES MODELS ==================")
    for model_name, model_fn in model_factories.items():
        print(f"\n>>> Running LOSO Evaluation for {model_name.upper()}...", flush=True)
        t_start_model = time.time()
        
        oof_y_true = []
        oof_y_pred = []
        oof_probs = []
        sub_metrics = []
        infer_latencies = []
        
        for sub in unique_subs:
            train_mask = (subjects != sub)
            test_mask = (subjects == sub)
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Channel-wise normalization (fit on train, transform test)
            mean = X_train.mean(axis=(0, 2), keepdims=True) # shape (1, 6, 1)
            std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            
            X_train_norm = (X_train - mean) / std
            X_test_norm = (X_test - mean) / std
            
            train_ds = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            test_ds = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            # Loss weighting
            pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Reset seed per fold for reproducibility
            torch.manual_seed(42)
            model = model_fn().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            
            for epoch in range(EPOCHS):
                train_epoch(model, train_loader, optimizer, criterion, device)
                scheduler.step()
                
            probs, preds, lat_ms = evaluate(model, test_loader, device)
            infer_latencies.append(lat_ms)
            
            oof_y_true.extend(y_test)
            oof_y_pred.extend(preds)
            oof_probs.extend(probs)
            
            f1_s = f1_score(y_test, preds, zero_division=0)
            macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            rec_s = recall_score(y_test, preds, zero_division=0)
            prec_s = precision_score(y_test, preds, zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, preds)
            
            # FPR
            neg_mask = (y_test == 0)
            fpr = float((preds[neg_mask] == 1).sum() / max(neg_mask.sum(), 1))
            
            sub_metrics.append({
                'model': model_name,
                'subject': sub,
                'f1_stress': float(f1_s),
                'macro_f1': float(macro_f1),
                'recall_stress': float(rec_s),
                'precision_stress': float(prec_s),
                'bal_acc': float(bal_acc),
                'fpr': float(fpr),
                'latency_ms': float(lat_ms)
            })
            per_sub_records.append(sub_metrics[-1])
            
        oof_y_true = np.array(oof_y_true)
        oof_y_pred = np.array(oof_y_pred)
        oof_probs = np.array(oof_probs)
        
        pooled_f1_stress = f1_score(oof_y_true, oof_y_pred, zero_division=0)
        pooled_macro_f1 = f1_score(oof_y_true, oof_y_pred, average='macro', zero_division=0)
        pooled_bal_acc = balanced_accuracy_score(oof_y_true, oof_y_pred)
        pooled_acc = accuracy_score(oof_y_true, oof_y_pred)
        pooled_auroc = roc_auc_score(oof_y_true, oof_probs)
        pooled_ap = average_precision_score(oof_y_true, oof_probs)
        pooled_recall = recall_score(oof_y_true, oof_y_pred, zero_division=0)
        pooled_prec = precision_score(oof_y_true, oof_y_pred, zero_division=0)
        
        oof_neg = (oof_y_true == 0)
        pooled_fpr = float((oof_y_pred[oof_neg] == 1).sum() / max(oof_neg.sum(), 1))
        
        # S7 specific metrics
        s7_rec = next(r for r in sub_metrics if r['subject'] == 'S7')
        
        mean_macro_f1 = np.mean([r['macro_f1'] for r in sub_metrics])
        mean_stress_f1 = np.mean([r['f1_stress'] for r in sub_metrics])
        mean_lat = np.mean(infer_latencies)
        
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        elapsed_s = time.time() - t_start_model
        
        summary_records.append({
            'model': model_name,
            'pooled_macro_f1': float(pooled_macro_f1),
            'mean_macro_f1': float(mean_macro_f1),
            'pooled_stress_f1': float(pooled_f1_stress),
            'mean_stress_f1': float(mean_stress_f1),
            'pooled_bal_acc': float(pooled_bal_acc),
            'pooled_auroc': float(pooled_auroc),
            'pooled_ap': float(pooled_ap),
            'pooled_recall': float(pooled_recall),
            'pooled_precision': float(pooled_prec),
            'pooled_fpr': float(pooled_fpr),
            's7_fpr': float(s7_rec['fpr']),
            's7_stress_f1': float(s7_rec['f1_stress']),
            'latency_ms_per_window': float(mean_lat),
            'peak_vram_mb': float(peak_vram_mb),
            'elapsed_sec': float(elapsed_s)
        })
        
        print(f"[{model_name.upper():6s}] Macro F1={pooled_macro_f1:.4f} | Stress F1={pooled_f1_stress:.4f} | "
              f"AUROC={pooled_auroc:.4f} | AP={pooled_ap:.4f} | FPR={pooled_fpr:.4f} | S7_FPR={s7_rec['fpr']:.4f} | "
              f"Lat={mean_lat:.2f}ms | VRAM={peak_vram_mb:.1f}MB ({elapsed_s:.1f}s)", flush=True)
              
    df_summary = pd.DataFrame(summary_records)
    df_sub = pd.DataFrame(per_sub_records)
    
    df_summary.to_csv(OUT_DIR / 'dl_summary.csv', index=False)
    df_sub.to_csv(OUT_DIR / 'dl_per_subject.csv', index=False)
    print(f"\nAll DL models evaluated. Results saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
