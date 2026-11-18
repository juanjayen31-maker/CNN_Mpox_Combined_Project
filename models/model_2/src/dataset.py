import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def prepare_dataset():
    # Download dataset Kaggle
    dataset_path = kagglehub.dataset_download(
        "joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"
    )

    print("Dataset berhasil didownload!")
    print("Lokasi dataset:", dataset_path)

    # Melihat struktur folder dataset
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, "").count(os.sep)

        if level > 4:
            dirs[:] = []
            continue

        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")

        for file in files[:5]:
            print(f"{indent}    {file}")

    # Menentukan lokasi folder dataset
    original_path = os.path.join(
        dataset_path,
        "Original Images",
        "Original Images",
        "FOLDS"
    )

    augmented_path = os.path.join(
        dataset_path,
        "Augmented Images",
        "Augmented Images",
        "FOLDS_AUG"
    )

    print("Original Images Path:", original_path)
    print("Augmented Images Path:", augmented_path)

    print("\nFold Original:")
    print(os.listdir(original_path))

    print("\nFold Augmented:")
    print(os.listdir(augmented_path))

    # Preprocessing gambar
    image_size = 100

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    print("Preprocessing berhasil dibuat!")

    return original_path, augmented_path, train_transform, test_transform
