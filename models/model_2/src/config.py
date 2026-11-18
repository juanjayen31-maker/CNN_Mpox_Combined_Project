import os
import torch

image_size = 100
num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_folds = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(BASE_DIR, "Hasil_5_Fold_Model2")
