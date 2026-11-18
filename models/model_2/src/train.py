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

from src.model import ModifiedCNN


def run_training(
    original_path,
    augmented_path,
    train_transform,
    test_transform,
    device,
    num_epochs,
    batch_size,
    learning_rate,
    results_path
):
    # ==========================================
    # TRAINING DAN EVALUASI OTOMATIS 5-FOLD - MODEL 2
    # ==========================================

    all_fold_results = []

    for fold_number in range(1, 6):

        fold = f"fold{fold_number}"

        print("\n" + "=" * 55)
        print(f"MEMULAI {fold.upper()}")
        print("=" * 55)

        # 1. Membaca dataset
        original_train_path = os.path.join(
            original_path, fold, "Train"
        )

        augmented_train_path = os.path.join(
            augmented_path, fold + "_AUG", "Train"
        )

        valid_path = os.path.join(
            original_path, fold, "Valid"
        )

        test_path = os.path.join(
            original_path, fold, "Test"
        )

        original_train = datasets.ImageFolder(
            original_train_path,
            transform=train_transform
        )

        augmented_train = datasets.ImageFolder(
            augmented_train_path,
            transform=train_transform
        )

        valid_dataset = datasets.ImageFolder(
            valid_path,
            transform=test_transform
        )

        test_dataset = datasets.ImageFolder(
            test_path,
            transform=test_transform
        )

        # Memastikan urutan kelas sama
        assert original_train.class_to_idx == augmented_train.class_to_idx
        assert original_train.class_to_idx == valid_dataset.class_to_idx
        assert original_train.class_to_idx == test_dataset.class_to_idx

        class_names = original_train.classes
        num_classes = len(class_names)

        # Menggabungkan data training
        train_dataset = ConcatDataset([
            original_train,
            augmented_train
        ])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        print("Original Train:", len(original_train))
        print("Augmented Train:", len(augmented_train))
        print("Total Training:", len(train_dataset))
        print("Validation:", len(valid_dataset))
        print("Testing:", len(test_dataset))

        # 2. Membuat model baru
        model = ModifiedCNN(num_classes=num_classes).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        best_valid_accuracy = -1
        best_model_state = None

        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []

        # 3. Training
        for epoch in range(num_epochs):

            model.train()

            running_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total

            # 4. Validation
            model.eval()

            valid_loss = 0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():

                for images, labels in valid_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs, 1)

                    valid_total += labels.size(0)
                    valid_correct += (
                        predicted == labels
                    ).sum().item()

            epoch_valid_loss = valid_loss / valid_total
            epoch_valid_acc = valid_correct / valid_total

            train_losses.append(epoch_train_loss)
            valid_losses.append(epoch_valid_loss)

            train_accuracies.append(epoch_train_acc)
            valid_accuracies.append(epoch_valid_acc)

            # Menyimpan model terbaik
            if epoch_valid_acc > best_valid_accuracy:

                best_valid_accuracy = epoch_valid_acc

                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | "
                f"Valid Loss: {epoch_valid_loss:.4f} | "
                f"Valid Acc: {epoch_valid_acc:.4f}"
            )

        # Memuat bobot terbaik
        model.load_state_dict(best_model_state)

        print("\nTraining selesai!")
        print(
            "Best Validation Accuracy:",
            round(best_valid_accuracy, 4)
        )

        # 5. Menyimpan grafik training
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(train_losses, label="Training Loss")
        axes[0].plot(valid_losses, label="Validation Loss")
        axes[0].set_title(f"Loss - {fold.upper()}")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(train_accuracies, label="Training Accuracy")
        axes[1].plot(valid_accuracies, label="Validation Accuracy")
        axes[1].set_title(f"Accuracy - {fold.upper()}")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                results_path,
                f"{fold}_training_graph.png"
            )
        )

        plt.show()
        plt.close()

        # 6. Testing
        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)

                outputs = model(images)

                probabilities = torch.softmax(outputs, dim=1)

                _, predictions = torch.max(outputs, 1)

                all_labels.extend(labels.numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 7. Menghitung metrik
        accuracy = accuracy_score(
            all_labels, all_predictions
        )

        precision = precision_score(
            all_labels, all_predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            all_labels, all_predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            all_labels, all_predictions,
            average="macro",
            zero_division=0
        )

        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities,
            multi_class="ovr",
            average="macro"
        )

        print(f"\nHASIL EVALUASI TESTING {fold.upper()}")
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1-Score :", round(f1, 4))
        print("ROC-AUC  :", round(roc_auc, 4))

        # 8. Classification Report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0
        )

        print("\nClassification Report:")
        print(report)

        with open(
            os.path.join(results_path, f"{fold}_report.txt"),
            "w"
        ) as file:
            file.write(report)

        # 9. Confusion Matrix
        cm = confusion_matrix(
            all_labels,
            all_predictions,
            labels=list(range(num_classes))
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title(f"Confusion Matrix - {fold.upper()}")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                results_path,
                f"{fold}_confusion_matrix.png"
            )
        )

        plt.show()
        plt.close()

        # 10. Menyimpan hasil metrik fold
        all_fold_results.append({
            "Fold": fold.upper(),
            "Best Validation Accuracy": best_valid_accuracy,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        print(f"{fold.upper()} BERHASIL DISELESAIKAN!")

    # ==========================================
    # 11. RATA-RATA HASIL 5-FOLD
    # ==========================================

    results_df = pd.DataFrame(all_fold_results)

    print("\n" + "=" * 55)
    print("HASIL EVALUASI SELURUH FOLD")
    print("=" * 55)

    print(results_df.to_string(index=False))

    average_results = results_df[
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ].mean()

    print("\nRATA-RATA HASIL 5-FOLD:")
    print(average_results.round(4))

    # Simpan tabel
    results_df.to_csv(
        os.path.join(results_path, "hasil_semua_fold.csv"),
        index=False
    )

    average_results.to_csv(
        os.path.join(results_path, "rata_rata_5_fold.csv")
    )

    print("\nSEMUA FOLD BERHASIL DISELESAIKAN!")
    print("Hasil tersimpan di:", results_path)
