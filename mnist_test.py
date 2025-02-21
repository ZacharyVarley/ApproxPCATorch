"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT

In this file we test stochastic PCA implementations on the MNIST dataset and
compare the learned subspaces to the true subspace, and vizualize example
reconstructions for an entry from each class.
"""

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import time

from pca_krasulina import KrasulinaPCA
from pca_oja_naive import OjaPCA
from pca_oja_exp import OjaPCAExp
from pca_oja_rop import OjaPCAROP

# Set random seed for reproducibility
torch.manual_seed(0)

# Load and preprocess MNIST (put it outside of the repo (parent dir))
trainset = datasets.MNIST(root="../data", train=True, download=True)
data_matrix = trainset.data.view(-1, 28 * 28).float()
# Normalize each image to [0,1] range
data_matrix = (data_matrix - torch.min(data_matrix, dim=1, keepdim=True)[0]) / (
    torch.max(data_matrix, dim=1, keepdim=True)[0]
    - torch.min(data_matrix, dim=1, keepdim=True)[0]
)

# Hyperparameters
k = 100  # number of components
b_size = 1024  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_matrix = data_matrix.to(device)

# Initialize all PCA methods
methods = {
    "Krasulina": KrasulinaPCA(
        n_features=784,
        n_components=k,
        base_eta=10.0,
    ).to(device),
    "Oja (Set LR)": OjaPCA(
        n_features=784,
        n_components=k,
        eta=0.005,
    ).to(device),
    "Oja (Exp LR)": OjaPCAExp(
        n_features=784,
        n_components=k,
        total_steps=60000 // b_size,
        initial_eta=1.0,
        final_eta=1e-4,
    ).to(device),
    "Oja (ROP LR)": OjaPCAROP(
        n_features=784,
        n_components=k,
        initial_eta=0.5,
        factor=0.8,
        patience=4,
        threshold=1e-4,
        min_eta=1e-6,
    ).to(device),
}

# Training results storage
results = {name: {"time": 0.0, "final_mse": 0.0} for name in methods.keys()}

# Training loop for each method
for name, pca in methods.items():
    print(f"\nTraining with {name}...")
    start = time.time()
    # One epoch of training
    for _ in range(1):  # change for multiple passes
        shuffled_data = data_matrix[torch.randperm(data_matrix.size(0))]
        for i in range(0, len(shuffled_data) - b_size, b_size):
            batch = shuffled_data[i : i + b_size]
            if len(batch) < b_size:
                # This line means we use up to an extra partial batch over 1 pass
                batch = torch.cat([batch, shuffled_data[: b_size - len(batch)]], dim=0)

            error = pca(batch) if hasattr(pca, "forward") else None

            if i % 1000 == 0 and error is not None:
                print(f"Batch {i}, MSE: {error:.8f}")
    torch.cuda.synchronize()
    results[name]["time"] = time.time() - start

# Compare with exact SVD
print("\nComputing exact SVD...")
start = time.time()
U, S, V = torch.svd(data_matrix.T)
torch.cuda.synchronize()
svd_time = time.time() - start
print(f"SVD time: {svd_time:.4f} seconds")
U_k = U[:, :k].mT

# Calculate all reconstructions upfront and store them
print("\nComputing reconstructions...")
recon_svd = (data_matrix @ U_k.T) @ U_k
svd_mse = torch.mean((data_matrix - recon_svd) ** 2).item()
print(f"SVD MSE: {svd_mse:.8f}")

all_reconstructions = {"SVD": recon_svd.cpu()}
all_components = {"SVD": U_k.cpu()}

for name, pca in methods.items():
    recon = pca.inverse_transform(pca.transform(data_matrix))
    mse = torch.mean((data_matrix - recon) ** 2).item()
    results[name]["final_mse"] = mse
    print(f"{name} MSE: {mse:.8f}")
    print(f"{name} Time: {results[name]['time']:.4f} seconds")

    # Store reconstructions and components
    all_reconstructions[name] = recon.cpu()
    components = pca.get_components()

    # adjust the sign of the components to align with SVD
    components *= torch.sign(
        torch.sum(components, dim=1, keepdim=True) * torch.sum(U_k, dim=1, keepdim=True)
    )
    all_components[name] = components.cpu()

# Move data to CPU for plotting
data_matrix = data_matrix.cpu()

# Figure 1: Components Visualization
print("\nPlotting components...")
n_methods = len(methods) + 1  # +1 for SVD
fig1, axs1 = plt.subplots(n_methods, 10, figsize=(15, 2 * n_methods))
plt.suptitle(f"First/Last 5 PCA Components For K={k}", fontsize=16)

max_val = 0.1  # For component visualization

# Plot all components
for row, (name, components) in enumerate(all_components.items()):
    for i in range(10):
        comp_idx = i if i <= 4 else k - (10 - i)  # First 5 and last 5 components
        axs1[row, i].imshow(
            components[comp_idx].view(28, 28),
            cmap="gray",
            vmin=-max_val,
            vmax=max_val,
        )
        axs1[row, i].axis("off")
        axs1[row, i].set_title(f"{name} {comp_idx+1}")

plt.tight_layout()
fig1.savefig("mnist_kpca_components.png", bbox_inches="tight", dpi=300)
plt.clf()

# Figure 2: Reconstructions
print("\nPlotting reconstructions...")
n_methods = len(methods) + 1  # +1 for SVD
fig2, axs2 = plt.subplots(n_methods + 1, 10, figsize=(15, 2 * (n_methods + 1)))
plt.suptitle(f"Original Images and Reconstructions for K={k}", fontsize=16)

# pluck out an example from each class
example_indices = []
for i in range(10):
    example_indices.append(
        torch.where(trainset.targets == i)[0][0]
    )  # Get the first occurrence of each class
example_indices = torch.tensor(example_indices)

# Plot original images in first row
for i, index in enumerate(example_indices):
    axs2[0, i].imshow(data_matrix[index].view(28, 28), cmap="gray", vmin=0, vmax=1)
    axs2[0, i].axis("off")
    axs2[0, i].set_title("Original")

# Plot reconstructions and differences
for row, (name, recon) in enumerate(all_reconstructions.items(), start=1):
    for i, index in enumerate(example_indices):
        # Plot reconstruction
        img = recon[index].view(28, 28)
        axs2[row, i].imshow(img, cmap="gray", vmin=0, vmax=1)

        axs2[row, i].axis("off")
        axs2[row, i].set_title(f"{name}")

plt.tight_layout()
fig2.savefig("mnist_kpca_reconstructions.png", bbox_inches="tight", dpi=300)
plt.clf()

# Figure 3: Reconstruction Errors
print("\nPlotting reconstruction errors...")
n_methods = len(methods) + 1  # +1 for SVD
fig2, axs2 = plt.subplots(n_methods + 1, 10, figsize=(15, 2 * (n_methods + 1)))
plt.suptitle("Reconstruction Errors", fontsize=16)

# pluck out an example from each class
example_indices = []
for i in range(10):
    example_indices.append(
        torch.where(trainset.targets == i)[0][0]
    )  # Get the first occurrence of each class
example_indices = torch.tensor(example_indices)

# Plot original images in first row
for i, index in enumerate(example_indices):
    axs2[0, i].imshow(data_matrix[index].view(28, 28), cmap="gray", vmin=0, vmax=1)
    axs2[0, i].axis("off")
    axs2[0, i].set_title("Original")

# Plot reconstructions and differences
for row, (name, recon) in enumerate(all_reconstructions.items(), start=1):
    for i, index in enumerate(example_indices):
        # Plot reconstruction
        img = recon[index].view(28, 28)
        # axs2[row, i].imshow(img, cmap="gray", vmin=0, vmax=1)

        # Overlay log-difference
        diff = torch.log10(torch.abs(img - data_matrix[index].view(28, 28)) + 1e-10)
        im = axs2[row, i].imshow(
            diff,
            cmap="viridis",
            alpha=1.0,
            vmin=-3,
            vmax=0,
        )

        axs2[row, i].axis("off")
        axs2[row, i].set_title(f"{name}")

        # # Add colorbar only for first image in row
        # if i == 0:
        #     plt.colorbar(im, ax=axs2[row, i], label="log10|diff|")

# put a giant colorbar on the right
cbar_ax = fig2.add_axes([1.02, 0.15, 0.03, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label("Recon log10|diff|", rotation=270, labelpad=15, fontsize=15)

plt.tight_layout()
fig2.savefig("mnist_kpca_reconstructions_errors.png", bbox_inches="tight", dpi=300)

# Print final results table
print("\nFinal Results:")
print(f"{'Method':<15} {'Time (s)':<10} {'MSE':<10}")
print("-" * 35)
print(f"{'SVD':<15} {svd_time:<10.4f} {svd_mse:<10.8f}")
for name, result in results.items():
    print(f"{name:<15} {result['time']:<10.4f} {result['final_mse']:<10.8f}")
