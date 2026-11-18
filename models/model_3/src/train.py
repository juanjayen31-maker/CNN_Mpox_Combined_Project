import os
import time
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DEVICE, NUM_WORKERS, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    ASSETS_DIR, MODEL_SAVE_DIR, TRAIN_OUTPUT_DIR
)
from src.dataset import CLAHETransform, SkinLesionDataset, get_transforms, get_actual_fold_path
from src.model import load_eva02_backbone, build_classifier_head


def save_preprocessing_sample(orig_dir):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    sample_fold_path = get_actual_fold_path(orig_dir, 1)
    sample_ds = SkinLesionDataset(os.path.join(sample_fold_path, "Train"), CLASS_NAMES)

    if len(sample_ds) > 0:
        sample_img_path = sample_ds.image_paths[0]
        raw_img = Image.open(sample_img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        clahe_img = CLAHETransform(clip_limit=2.0)(raw_img)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(raw_img)
        ax[0].set_title("Citra Asli (Original)")
        ax[0].axis("off")

        ax[1].imshow(clahe_img)
        ax[1].set_title("Setelah LAB-CLAHE")
        ax[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, "preprocessing_comparison.png"), dpi=200)
        plt.close(fig)


def extract_embeddings(backbone, loader, desc="Ekstraksi Fitur"):
    feats_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(DEVICE)
            emb = backbone(images)
            feats_list.append(emb.cpu())
            labels_list.append(labels)
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


def run_training(orig_dir, aug_dir):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    _, eval_tf = get_transforms()

    backbone, selected_model = load_eva02_backbone()
    print(f"\n[TRAINING] Memuat Feature Extractor: {selected_model} pada {DEVICE}...")

    for fold in range(1, 6):
        start_t = time.time()
        print(f"\n{'='*60}\nTRAINING FOLD {fold} - MODEL: {selected_model}\n{'='*60}")
        orig_fold_path = get_actual_fold_path(orig_dir, fold)
        aug_fold_path  = get_actual_fold_path(aug_dir, fold)

        # Data Train (Original + Augmented) & Valid (Original)
        ds_orig_train = SkinLesionDataset(os.path.join(orig_fold_path, "Train"), CLASS_NAMES, transform=eval_tf)
        ds_aug_train  = SkinLesionDataset(os.path.join(aug_fold_path, "Train"), CLASS_NAMES, transform=eval_tf)
        train_dataset = ConcatDataset([ds_orig_train, ds_aug_train])
        valid_dataset = SkinLesionDataset(os.path.join(orig_fold_path, "Valid"), CLASS_NAMES, transform=eval_tf)

        print(f"Jumlah sampel Fold {fold} -> Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

        X_train, y_train = extract_embeddings(backbone, train_loader, desc=f"Fold {fold} [Ekstrak Train]")
        X_valid, y_valid = extract_embeddings(backbone, valid_loader, desc=f"Fold {fold} [Ekstrak Valid]")

        feat_mean = X_train.mean(dim=0, keepdim=True)
        feat_std  = X_train.std(dim=0, keepdim=True) + 1e-6
        X_train = (X_train - feat_mean) / feat_std
        X_valid = (X_valid - feat_mean) / feat_std

        class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train.numpy())
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        head = build_classifier_head(X_train.shape[1])
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

        head_epochs = 25
        best_val_loss = float("inf")
        best_head_weights = None
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        train_emb_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

        for epoch in range(1, head_epochs + 1):
            head.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for bx, by in train_emb_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = head(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * bx.size(0)
                tr_correct += (out.argmax(dim=1) == by).sum().item()
                tr_total += by.size(0)

            head.eval()
            with torch.no_grad():
                vx, vy = X_valid.to(DEVICE), y_valid.to(DEVICE)
                vout = head(vx)
                val_loss = criterion(vout, vy).item()
                val_acc = (vout.argmax(dim=1) == vy).float().mean().item()

            history["train_loss"].append(tr_loss / tr_total)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_correct / tr_total)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_head_weights = copy.deepcopy(head.state_dict())

        # Simpan bobot model terbaik & parameter normalisasi untuk testing
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"best_eva02_head_fold_{fold}.pth")
        torch.save({
            "in_features": X_train.shape[1],
            "head_state_dict": best_head_weights,
            "feat_mean": feat_mean,
            "feat_std": feat_std
        }, checkpoint_path)

        # Simpan kurva training ke folder outputs/training/
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_acc"], label="Train Accuracy")
        axes[0].plot(history["val_acc"], label="Validation Accuracy")
        axes[0].set_title(f"EVA-02 Fold {fold} - Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history["train_loss"], label="Train Loss")
        axes[1].plot(history["val_loss"], label="Validation Loss")
        axes[1].set_title(f"EVA-02 Fold {fold} - Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(TRAIN_OUTPUT_DIR, f"training_curve_fold_{fold}.png"), dpi=200)
        plt.close(fig)

        elapsed = time.time() - start_t
        print(f"--> Training Fold {fold} selesai dalam {elapsed:.1f} detik | Bobot disimpan ke: {checkpoint_path}")