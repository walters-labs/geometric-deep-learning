"""
2d-SO2-invariant-cnn-mnist.py

A 2D rotation-invariant CNN using e2cnn trained on rotated MNIST.

- Uses discrete rotations matching the group order N.
- Applies GroupPooling for invariance.
- Trains on MNIST augmented with discrete rotations.
- Tests invariance on random test samples.
- Optionally visualizes inputs.

Requirements:
    pip install torch torchvision e2cnn matplotlib numpy pillow

Run:
    python 2d-SO2-invariant-cnn-mnist.py --N 8 --epochs 5 --viz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from e2cnn import gspaces
from e2cnn import nn as enn
from torchvision import datasets, transforms
import numpy as np
import argparse
from PIL import Image
import random


# ---------------------------------------------------------------------
# Custom transform: discrete rotation by multiples of 360/N degrees
# ---------------------------------------------------------------------
class DiscreteRotation:
    def __init__(self, N=4):
        self.N = N
        self.angles = [360.0 * k / N for k in range(N)]

    def __call__(self, img):
        angle = random.choice(self.angles)
        return img.rotate(angle, resample=Image.BILINEAR)


# ---------------------------------------------------------------------
# Model: equivariant + invariant CNN
# ---------------------------------------------------------------------
class InvariantEquivariantNet(nn.Module):
    def __init__(self, N=4, n_classes=10):
        super().__init__()
        self.N = N
        self.gspace = gspaces.Rot2dOnR2(N=N)

        in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        out1 = enn.FieldType(self.gspace, [self.gspace.regular_repr])
        out2 = enn.FieldType(self.gspace, [self.gspace.regular_repr])

        self.block1 = enn.R2Conv(in_type, out1, kernel_size=7, padding=3, bias=False)
        self.relu1 = enn.ReLU(out1, inplace=True)
        self.block2 = enn.R2Conv(out1, out2, kernel_size=5, padding=2, bias=False)
        self.relu2 = enn.ReLU(out2, inplace=True)

        self.gpool = enn.GroupPooling(out2)
        self.fc = nn.Linear(1, n_classes)

        self.in_type = in_type
        self.out_type = out2

    def forward(self, geo_x):
        x = self.relu1(self.block1(geo_x))
        x = self.relu2(self.block2(x))
        x = self.gpool(x)
        t = x.tensor.mean(dim=[2, 3])
        return self.fc(t)


# ---------------------------------------------------------------------
# Train function with batch-level updates
# ---------------------------------------------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0
    total_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        geo_data = enn.GeometricTensor(data, model.in_type)

        optimizer.zero_grad()
        output = model(geo_data)
        loss = criterion(output, F.one_hot(target, num_classes=output.shape[1]).float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            avg_loss = running_loss / batch_idx
            print(f"[Train] Epoch {epoch} Batch {batch_idx}/{total_batches} — Avg Loss: {avg_loss:.4f}")

    avg_loss = running_loss / total_batches
    print(f"[Train] Epoch {epoch} Completed — Avg Loss: {avg_loss:.4f}")


# ---------------------------------------------------------------------
# Test function with batch-level accuracy updates
# ---------------------------------------------------------------------
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    total_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader, 1):
            data, target = data.to(device), target.to(device)
            geo_data = enn.GeometricTensor(data, model.in_type)
            output = model(geo_data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                acc_so_far = correct / total * 100
                print(f"[Test] Batch {batch_idx}/{total_batches} — Accuracy so far: {acc_so_far:.2f}%")

    acc = correct / total
    print(f"[Test] Final Accuracy: {acc*100:.2f}%")
    return acc


# ---------------------------------------------------------------------
# Invariance tests on single sample
# ---------------------------------------------------------------------
def test_invariance(model, device, N, noise=0.01, viz=False):
    model.eval()
    # Use a sample MNIST digit from test set
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    img, label = mnist_test[0]  # first test sample, tensor [1,28,28]
    img = img.squeeze(0).numpy()

    # Resize to 64x64 for model input
    img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR)
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    # Add noise
    img_np += noise * np.random.randn(*img_np.shape)
    img_np = np.clip(img_np, 0, 1)

    # Base output
    X = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    geo_X = enn.GeometricTensor(X, model.in_type)
    base_out = model(geo_X).detach()

    print("\n=== Invariance Tests on MNIST sample ===")
    angles = [360.0 * k / N for k in range(N)]
    for angle in angles:
        img_r = pil_rotate(img_np, angle=angle)
        Xr = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0).to(device)
        geo_Xr = enn.GeometricTensor(Xr, model.in_type)
        out_r = model(geo_Xr).detach()

        diff = (out_r - base_out).abs()
        print(f"Rotation {angle:6.1f}° | max diff: {diff.max():.3e} | mean diff: {diff.mean():.3e}")

        tol = 1e-3
        if diff.max() > tol:
            print(f"⚠️ Warning: Invariance test failed at rotation {angle}°")

    if viz:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Original (label {label})")
        plt.imshow(img_np, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title(f"Rotated {angles[1]:.1f}°")
        plt.imshow(pil_rotate(img_np, angles[1]), cmap="gray")
        plt.axis("off")
        plt.show()


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SO(2) invariant CNN on rotated MNIST.")
    parser.add_argument("--N", type=int, default=8, help="Number of discrete rotations (default 8)")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--viz", action="store_true", help="Visualize sample images and rotations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transform with discrete rotation augmentation
    train_transform = transforms.Compose([
        DiscreteRotation(N=args.N),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    model = InvariantEquivariantNet(N=args.N).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader)

    # Invariance test on one sample
    test_invariance(model, device, args.N, noise=0.01, viz=args.viz)


if __name__ == "__main__":
    main()
