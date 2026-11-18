import os
import platform
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_WINDOWS = platform.system() == "Windows"
NUM_WORKERS = 0 if IS_WINDOWS else 2  # Menghindari error multiprocessing di Windows VS Code

MODEL_NAME  = "eva02_tiny_patch14_224.mim_in22k_ft_in22k_in1k"
IMG_SIZE    = 224
BATCH_SIZE  = 16 if not torch.cuda.is_available() else 32
EPOCHS      = 15
LR          = 5e-5
NUM_CLASSES = 6
CLASS_NAMES = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]

# Path relatif terhadap folder utama proyek
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")
ASSETS_DIR       = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR   = os.path.join(OUTPUT_DIR, "models")
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training")
TEST_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, "testing")