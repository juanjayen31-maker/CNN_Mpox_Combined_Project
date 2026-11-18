import torch.nn as nn
import timm
from src.config import DEVICE, NUM_CLASSES


def load_eva02_backbone():
    available_eva = timm.list_models("*eva02_tiny*", pretrained=True)
    selected_model = (
        "eva02_tiny_patch14_224"
        if "eva02_tiny_patch14_224" in available_eva
        else (available_eva[0] if available_eva else "eva02_tiny_patch14_224")
    )
    backbone = timm.create_model(selected_model, pretrained=True, num_classes=0)
    backbone.to(DEVICE)
    backbone.eval()
    return backbone, selected_model


def build_classifier_head(in_features):
    head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    ).to(DEVICE)
    return head