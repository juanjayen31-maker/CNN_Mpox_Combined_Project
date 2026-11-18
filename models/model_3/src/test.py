import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from src.config import (
    DEVICE, NUM_WORKERS, NUM_CLASSES, CLASS_NAMES,
    MODEL_SAVE_DIR, TEST_OUTPUT_DIR
)
from src.dataset import SkinLesionDataset, get_transforms, get_actual_fold_path
from src.model import load_eva02_backbone, build_classifier_head
from src.train import extract_embeddings


def run_testing(orig_dir):
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    _, eval_tf = get_transforms()

    backbone, selected_model = load_eva02_backbone()
    print(f"\n[TESTING] Memuat Feature Extractor: {selected_model} pada {DEVICE}...")

    metrics_log = {
        "Fold": [], "Accuracy": [], "Precision": [],
        "Recall": [], "F1-Score": [], "ROC-AUC": []
    }
    aggregate_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for fold in range(1, 6):
        orig_fold_path = get_actual_fold_path(orig_dir, fold)
        test_dataset = SkinLesionDataset(os.path.join(orig_fold_path, "Test"), CLASS_NAMES, transform=eval_tf)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"best_eva02_head_fold_{fold}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model hasil training tidak ditemukan: {checkpoint_path}. Jalankan training terlebih dahulu!"
            )

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        feat_mean = checkpoint["feat_mean"]
        feat_std  = checkpoint["feat_std"]

        X_test, y_test = extract_embeddings(backbone, test_loader, desc=f"Fold {fold} [Ekstrak Test]")
        X_test = (X_test - feat_mean) / feat_std

        head = build_classifier_head(checkpoint["in_features"])
        head.load_state_dict(checkpoint["head_state_dict"])
        head.eval()

        with torch.no_grad():
            tx = X_test.to(DEVICE)
            probs = torch.softmax(head(tx), dim=1).cpu().numpy()

        y_true = y_test.numpy()
        y_pred = np.argmax(probs, axis=1)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc  = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        cm   = confusion_matrix(y_true, y_pred)

        aggregate_cm += cm
        metrics_log["Fold"].append(f"Fold {fold}")
        metrics_log["Accuracy"].append(acc)
        metrics_log["Precision"].append(prec)
        metrics_log["Recall"].append(rec)
        metrics_log["F1-Score"].append(f1)
        metrics_log["ROC-AUC"].append(auc)

        print(f"--> Hasil Test Fold {fold} (Samples={len(test_dataset)}): Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    df_results = pd.DataFrame(metrics_log)
    mean_summary = pd.DataFrame({
        "Fold": ["Mean (5-Fold)"],
        "Accuracy": [df_results["Accuracy"].mean()],
        "Precision": [df_results["Precision"].mean()],
        "Recall": [df_results["Recall"].mean()],
        "F1-Score": [df_results["F1-Score"].mean()],
        "ROC-AUC": [df_results["ROC-AUC"].mean()]
    })
    df_final = pd.concat([df_results, mean_summary], ignore_index=True)

    print("\n" + "=" * 68)
    print("REKAP HASIL EVALUASI 5-FOLD CROSS VALIDATION (MODEL: EVA-02)")
    print("=" * 68)
    print(df_final.to_string(index=False))

    csv_path = os.path.join(TEST_OUTPUT_DIR, "hasil_evaluasi_eva02_5fold.csv")
    df_final.to_csv(csv_path, index=False)

    # Simpan Aggregate Confusion Matrix ke folder outputs/testing/
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(aggregate_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Aggregate Confusion Matrix - EVA-02 (5-Fold Test Set)")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, "confusion_matrix_5fold.png"), dpi=200)
    plt.close(fig_cm)
    print(f"\nSeluruh hasil testing dan Confusion Matrix tersimpan di:\n{TEST_OUTPUT_DIR}")