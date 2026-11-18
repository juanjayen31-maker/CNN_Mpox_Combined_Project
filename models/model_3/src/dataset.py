import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.config import IMG_SIZE


def find_dataset_folders(search_roots):
    orig_folds_dir = None
    aug_folds_dir = None

    for search_root in search_roots:
        if not search_root or not os.path.exists(search_root):
            continue
        for root, dirs, _ in os.walk(search_root):
            has_fold1 = any("fold" in d.lower() and "1" in d.lower() for d in dirs)
            if has_fold1:
                if "aug" in root.lower():
                    aug_folds_dir = root
                else:
                    orig_folds_dir = root
        if orig_folds_dir and aug_folds_dir:
            break

    return orig_folds_dir, aug_folds_dir


def get_actual_fold_path(base_folds_dir, fold_num):
    for name in os.listdir(base_folds_dir):
        name_low = name.lower()
        if "fold" in name_low and str(fold_num) in name_low:
            return os.path.join(base_folds_dir, name)
    raise FileNotFoundError(f"Fold {fold_num} tidak ditemukan di {base_folds_dir}")


class CLAHETransform:
    """
    Meningkatkan kontras lokal pada channel Lightness (L) di ruang warna LAB
    untuk mempertegas tekstur pustula/ruam tanpa merusak informasi warna kulit.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_clahe)


class SkinLesionDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Direktori tidak ditemukan: {root_dir}")

        existing_folders = {
            d.lower(): d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        }

        for idx, cls_name in enumerate(class_names):
            actual_cls_folder = existing_folders.get(cls_name.lower())
            if actual_cls_folder is None:
                continue
            cls_path = os.path.join(root_dir, actual_cls_folder)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
                for path in glob.glob(os.path.join(cls_path, ext)):
                    self.image_paths.append(path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    eva_mean = (0.48145466, 0.4578275, 0.40821073)
    eva_std  = (0.26862954, 0.26130258, 0.27577711)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        CLAHETransform(clip_limit=2.0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=eva_mean, std=eva_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        CLAHETransform(clip_limit=2.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=eva_mean, std=eva_std)
    ])
    return train_transform, eval_transform