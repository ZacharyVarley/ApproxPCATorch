"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT

Oja batched update with ReduceLROnPlateau learning rate scheduler.

This module implements the batched version of Oja's update rule for PCA. Each
batch is used to update the estimates of the top-k eigenvectors of the
covariance matrix. Then a QR decomposition is performed to re-orthogonalize the
estimates. QR is O(mn^2) computed via Schwarz-Rutishauser for m x n matrices.

References:

Allen-Zhu, Zeyuan, and Yuanzhi Li. “First Efficient Convergence for Streaming
K-PCA: A Global, Gap-Free, and Near-Optimal Rate.” 2017 IEEE 58th Annual
Symposium on Foundations of Computer Science (FOCS), IEEE, 2017, pp. 487–92.
DOI.org (Crossref), https://doi.org/10.1109/FOCS.2017.51.

Tang, Cheng. “Exponentially Convergent Stochastic K-PCA without Variance
Reduction.” Advances in Neural Information Processing Systems, vol. 32, 2019.

"""

import torch
from torch import nn, Tensor


class ReduceLROnPlateau:
    def __init__(
        self,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-8,
        verbose=False,
    ):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better()

    def _init_is_better(self):
        """Initialize is_better function based on mode and threshold mode."""
        if self.mode not in {"min", "max"}:
            raise ValueError("mode " + self.mode + " is unknown!")

        if self.mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")

        if self.threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + self.threshold_mode + " is unknown!")

        if self.mode == "min":
            self.is_better = lambda a, best: (
                a < best - self.threshold
                if self.threshold_mode == "abs"
                else a < best * (1 - self.threshold)
            )
            self.mode_worse = float("inf")
        else:
            self.is_better = lambda a, best: (
                a > best + self.threshold
                if self.threshold_mode == "abs"
                else a > best * (1 + self.threshold)
            )
            self.mode_worse = -float("inf")

    def step(self, metrics, current_lr):
        """Returns whether to reduce the learning rate based on the current metric."""
        if self.best is None:
            self.best = metrics
            return current_lr

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            if self.cooldown_counter <= 0:
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr < current_lr - self.eps:
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
                    if self.verbose:
                        print(f"Reducing learning rate to {new_lr}")
                    return new_lr

        return current_lr


class OjaPCAROP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        initial_eta: float = 0.5,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_eta: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        use_oja_plus: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.eta = initial_eta
        self.use_oja_plus = use_oja_plus

        # Initialize parameters
        self.register_buffer("Q", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_eta,
            verbose=False,
        )

        # For Oja++
        if self.use_oja_plus:
            self.register_buffer(
                "initialized_cols", torch.zeros(n_components, dtype=torch.bool)
            )
            self.register_buffer("next_col_to_init", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> float:
        # Forward pass and reconstruction
        projection = x @ self.Q
        reconstruction = projection @ self.Q.T
        current_error = torch.mean((x - reconstruction) ** 2).item()

        # Update then Orthonormalize Q_t using QR decomposition
        self.Q.copy_(torch.linalg.qr(self.Q + self.eta * (x.T @ (projection)))[0])

        # Update learning rate based on error
        self.eta = self.scheduler.step(current_error, self.eta)

        # Update step counter
        self.step.add_(1)

        # For Oja++, gradually initialize columns
        if self.use_oja_plus and self.next_col_to_init < self.n_components:
            if self.step % (self.n_components // 2) == 0:
                self.Q[:, self.next_col_to_init] = torch.randn(
                    self.n_features, dtype=self.Q.dtype
                )
                self.initialized_cols[self.next_col_to_init] = True
                self.next_col_to_init.add_(1)

        return current_error

    def get_components(self) -> Tensor:
        return self.Q.T

    def transform(self, x: Tensor) -> Tensor:
        return x @ self.Q

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x @ self.Q.T
