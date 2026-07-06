"""
train.py
---------
Trains a convolutional denoising autoencoder on the MNIST dataset.

Steps performed:
    1. Download/load MNIST training and test data.
    2. For every batch, create a noisy version of the images.
    3. Feed the NOISY images into the model, and compare its output
       against the ORIGINAL CLEAN images using MSE loss.
    4. Save the best model (lowest validation loss) to disk.
    5. Plot a sample comparison: noisy vs denoised vs original.

Run with:
    python src/train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DenoisingAutoencoder
from utils import add_noise, imshow_grid

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = "../data"
MODEL_SAVE_PATH = "../models/denoising_autoencoder.pth"
OUTPUT_DIR = "../outputs"
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NOISE_FACTOR = 0.4

os.makedirs(MODEL_SAVE_PATH.rsplit("/", 1)[0], exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_dataloaders():
    """Downloads MNIST (if needed) and returns train/test DataLoaders."""
    transform = transforms.ToTensor()  # scales pixel values to [0, 1]

    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_test_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---------------- Training ----------------
        model.train()
        running_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            noisy_images = add_noise(images, NOISE_FACTOR).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)  # compare to CLEAN images
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---------------- Validation ----------------
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                noisy_images = add_noise(images, NOISE_FACTOR).to(device)
                outputs = model(noisy_images)
                loss = criterion(outputs, images)
                test_running_loss += loss.item() * images.size(0)

        test_loss = test_running_loss / len(test_loader.dataset)

        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}")

        # Save the best model so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved (test loss {test_loss:.6f}) to {MODEL_SAVE_PATH}")

    print("\nTraining complete.")
    visualize_results(model, test_loader, device)


def visualize_results(model, test_loader, device, n_images=10):
    """Shows/saves a comparison of noisy, denoised, and original images."""
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)
    noisy_images = add_noise(images, NOISE_FACTOR).to(device)

    with torch.no_grad():
        denoised_images = model(noisy_images)

    save_path = os.path.join(OUTPUT_DIR, "comparison.png")
    imshow_grid(
        [images[:n_images], noisy_images[:n_images], denoised_images[:n_images]],
        titles=["Original", "Noisy Input", "Denoised Output"],
        n=n_images,
        save_path=save_path,
    )


if __name__ == "__main__":
    train()
