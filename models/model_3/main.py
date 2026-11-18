# -*- coding: utf-8 -*-
"""main.py

Entry point utama proyek Study MonkeyPox Classification.
Terhubung langsung ke Kaggle (kagglehub) agar portabel dan dapat dijalankan
di perangkat mana pun tanpa bergantung pada path lokal spesifik.
"""

import os
import sys
import subprocess
import importlib


def ensure_packages():
    """Memastikan seluruh library utama (termasuk kagglehub) terinstal."""
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "cv2": "opencv-python-headless",
        "PIL": "pillow",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "sklearn": "scikit-learn",
        "kagglehub": "kagglehub",
    }
    missing = []
    for mod_name, pip_name in required.items():
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"Menginstal dependensi yang belum tersedia: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import random
import numpy as np
import torch
import kagglehub

from src.config import DEVICE, BATCH_SIZE, NUM_WORKERS
from src.dataset import find_dataset_folders
from src.train import save_preprocessing_sample, run_training
from src.test import run_testing


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything(42)
    print(f"Perangkat aktif : {DEVICE}")
    print(f"Batch Size      : {BATCH_SIZE} | Num Workers: {NUM_WORKERS}")

    # Mengambil dataset langsung dari Kaggle menggunakan kagglehub
    dataset_slug = "joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"
    print(f"Menghubungkan & mengunduh dataset dari Kaggle ({dataset_slug})...")
    kaggle_path = kagglehub.dataset_download(dataset_slug)

    orig_dir, aug_dir = find_dataset_folders([kaggle_path])

    print(f"Original Folds  : {orig_dir}")
    print(f"Augmented Folds : {aug_dir}")

    assert orig_dir is not None, "Folder Original Images tidak ditemukan di dataset Kaggle!"
    assert aug_dir is not None, "Folder Augmented Images tidak ditemukan di dataset Kaggle!"

    # Mode eksekusi: 'train', 'test', atau 'all' (default)
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode in ("all", "train"):
        save_preprocessing_sample(orig_dir)
        run_training(orig_dir, aug_dir)

    if mode in ("all", "test"):
        run_testing(orig_dir)


if __name__ == "__main__":
    main()