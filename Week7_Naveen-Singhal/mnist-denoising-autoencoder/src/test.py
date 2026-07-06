"""
test.py
--------
Loads a previously trained denoising autoencoder and runs it on fresh
MNIST test images to visualize denoising performance.

Run with:
    python src/test.py
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DenoisingAutoencoder
from utils import add_noise, imshow_grid

DATA_DIR = "../data"
MODEL_PATH = "../models/denoising_autoencoder.pth"
OUTPUT_DIR = "../outputs"
NOISE_FACTOR = 0.4
N_IMAGES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            f"Run 'python train.py' first to train and save a model."
        )

    # Load test data
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=N_IMAGES, shuffle=True)

    # Load trained model
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Loaded trained model successfully.")

    # Grab a batch and run inference
    images, _ = next(iter(test_loader))
    images = images.to(device)
    noisy_images = add_noise(images, NOISE_FACTOR).to(device)

    with torch.no_grad():
        denoised_images = model(noisy_images)

    save_path = os.path.join(OUTPUT_DIR, "test_comparison.png")
    imshow_grid(
        [images, noisy_images, denoised_images],
        titles=["Original", "Noisy Input", "Denoised Output"],
        n=N_IMAGES,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
