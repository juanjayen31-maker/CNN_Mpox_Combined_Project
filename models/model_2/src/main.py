import os

from src.config import (
    device, num_epochs, batch_size, learning_rate,
    num_folds, results_path
)
from src.dataset import prepare_dataset
from src.train import run_training


def main():
    # Menampilkan device yang digunakan
    print("Device yang digunakan:", device)

    if device.type == "cuda":
        import torch
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU tidak tersedia, menggunakan CPU.")

    # Pengaturan hyperparameter Model 2
    os.makedirs(results_path, exist_ok=True)

    print("Jumlah Epoch:", num_epochs)
    print("Batch Size:", batch_size)
    print("Learning Rate:", learning_rate)
    print("Folder hasil:", results_path)

    # Menyiapkan dataset dan preprocessing
    original_path, augmented_path, train_transform, test_transform = (
        prepare_dataset()
    )

    # Menjalankan training dan evaluasi otomatis 5-Fold
    run_training(
        original_path=original_path,
        augmented_path=augmented_path,
        train_transform=train_transform,
        test_transform=test_transform,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        results_path=results_path
    )


if __name__ == "__main__":
    main()
