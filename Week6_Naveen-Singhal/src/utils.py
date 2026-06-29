"""
utils.py
---------
Helper functions shared across training and testing scripts:
    - add_noise(): injects Gaussian noise into clean images
    - imshow_grid(): visualizes a row of images (used for before/after comparisons)
"""

import torch
import matplotlib.pyplot as plt


def add_noise(images, noise_factor=0.4):
    """
    Adds random Gaussian noise to a batch of images and clips the result
    back into the valid [0, 1] pixel range.

    Args:
        images (torch.Tensor): batch of clean images, shape (B, 1, 28, 28), values in [0,1]
        noise_factor (float): controls how strong the noise is (0 = no noise)

    Returns:
        torch.Tensor: noisy images, same shape as input, clipped to [0, 1]
    """
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    noisy_images = torch.clip(noisy_images, 0.0, 1.0)
    return noisy_images


def imshow_grid(images_list, titles, n=10, save_path=None):
    """
    Displays/saves rows of images for comparison (e.g. original vs noisy vs denoised).

    Args:
        images_list (list[torch.Tensor]): list of image batches, each shape (B, 1, 28, 28)
        titles (list[str]): row labels, one per entry in images_list
        n (int): number of example images (columns) to show per row
        save_path (str or None): if provided, saves the figure to this path instead of
                                  only displaying it
    """
    num_rows = len(images_list)
    fig, axes = plt.subplots(num_rows, n, figsize=(n * 1.2, num_rows * 1.4))

    for row in range(num_rows):
        imgs = images_list[row].cpu().detach()
        for col in range(n):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            img = imgs[col].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(titles[row], fontsize=10)
        # Add row title using the first subplot in the row
        first_ax = axes[row, 0] if num_rows > 1 else axes[0]
        first_ax.set_title(titles[row], loc="left", fontsize=11, pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved comparison figure to: {save_path}")
    plt.show()
