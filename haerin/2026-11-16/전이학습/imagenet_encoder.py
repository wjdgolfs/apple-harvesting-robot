"""Frozen TorchVision ResNet18 encoder shared by both apple experiments."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18


WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def build_encoder(weights_path, device):
    """Load the verified ImageNet-1K weights, then expose the 512 pooled features."""
    path = Path(weights_path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Expected TorchVision's ResNet18 IMAGENET1K_V1 weights at {path}")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.fc = torch.nn.Identity()
    return model.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def encode_batch(model, images_u8, device):
    """Encode already-sized RGB images. Input is NHWC uint8, output is CPU float32."""
    if isinstance(images_u8, (list, tuple)):
        images_u8 = np.stack([np.asarray(im) for im in images_u8])
    pixels = torch.as_tensor(images_u8)
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != torch.uint8:
        raise ValueError("Expected a batch of RGB images in NHWC uint8 format")
    batch = pixels.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
    mean = batch.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(RGB_STD).view(1, 3, 1, 1)
    features = model(batch.sub_(mean).div_(std))
    return features.to(device="cpu", dtype=torch.float32)
