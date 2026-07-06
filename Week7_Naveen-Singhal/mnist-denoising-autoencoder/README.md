# MNIST Image Denoising using Autoencoders

A convolutional autoencoder that learns to remove noise from handwritten digit
images (MNIST). The model is trained by feeding it **noisy** images and asking
it to reconstruct the **clean, original** images.

## How it works

1. Take a clean MNIST digit image (28x28 grayscale, pixel values 0-1).
2. Add random Gaussian noise to it.
3. Feed the **noisy** image into the autoencoder.
4. Compare the autoencoder's output to the **original clean** image using
   Mean Squared Error (MSE) loss.
5. Backpropagate and update the weights so the network gets better at
   stripping out the noise.

```
Clean Image --> [+ Noise] --> Noisy Image --> [Encoder] --> Compressed
                                                                  |
                                                              [Decoder]
                                                                  |
                                                          Denoised Image
                                                                  |
                                                  Loss = MSE(Denoised, Clean)
```

## Project structure

```
mnist-denoising-autoencoder/
├── requirements.txt        # Python dependencies
├── README.md                # This file
├── src/
│   ├── model.py              # Autoencoder architecture (CNN encoder/decoder)
│   ├── utils.py               # Noise injection + visualization helpers
│   ├── train.py                # Main training script
│   └── test.py                  # Run inference with a saved trained model
├── models/                  # Trained model weights get saved here (.pth)
├── outputs/                 # Saved comparison images get saved here (.png)
└── data/                    # MNIST dataset auto-downloads here
```

## Model architecture

**Encoder** (compresses 28x28 image down to a 16x7x7 representation):
- Conv2d(1 → 32, kernel=3) → ReLU → MaxPool(2)
- Conv2d(32 → 16, kernel=3) → ReLU → MaxPool(2)

**Decoder** (reconstructs back to 28x28):
- ConvTranspose2d(16 → 32, kernel=2, stride=2) → ReLU
- ConvTranspose2d(32 → 1, kernel=2, stride=2) → Sigmoid

**Loss function:** MSELoss (since we're comparing continuous pixel values,
this is a regression problem, not classification).

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (downloads MNIST automatically on first run)
cd src
python train.py

# 3. Test the trained model on fresh images
python test.py
```

Training for 20 epochs takes roughly 5-15 minutes on CPU, or under 2 minutes
on a GPU.

After training, check the `outputs/` folder for `comparison.png` — a visual
grid showing original, noisy, and denoised images side by side. The trained
model weights are saved in `models/denoising_autoencoder.pth`.

## Tuning

You can adjust these settings directly inside `src/train.py`:

| Variable | Description | Default |
|---|---|---|
| `NOISE_FACTOR` | How much noise is added (0 = none, 1 = very noisy) | 0.4 |
| `BATCH_SIZE` | Images per training batch | 64 |
| `NUM_EPOCHS` | Number of full passes over the training data | 20 |
| `LEARNING_RATE` | Adam optimizer learning rate | 1e-3 |
