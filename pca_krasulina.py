"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT

This module implements the batched version of the implicit Krasulina update rule
for top-k PCA component estimation in a streamed setting. Unfortunately, this
implementation has a k x k matrix inversion each step. I am unsure if the
Woodbury matrix identity can be used to reduce the cost of this inversion.

Unlike the reference below, I do not expect to be able to instantiate a d x d
PCA projection matrix P in memory. Instead I directly do SVD on C_cross.T at the
end to get the eigenvectors of the learned subspace.

References:

Amid, Ehsan, and Manfred K. Warmuth. “An Implicit Form of Krasulina’s k-PCA
Update without the Orthonormality Constraint.” Proceedings of the AAAI
Conference on Artificial Intelligence, vol. 34, no. 04, Apr. 2020, pp. 3179–86.
DOI.org (Crossref), https://doi.org/10.1609/aaai.v34i04.5715.

"""

import torch
from torch import Tensor


class KrasulinaPCA(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        base_eta: float = 10.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.base_eta = base_eta

        # Initialize parameters
        self.register_buffer("C", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer(
            "C_cross", torch.linalg.inv_ex(self.C.T @ self.C)[0] @ self.C.T
        )

    def forward(self, x: Tensor) -> None:
        # (n_obs, n_comp) = (n_obs, n_feat) @ (n_feat, n_comp)
        latent = x @ self.C_cross.T

        # (n_obs, n_feat) = (n_obs, n_feat) - (n_obs, n_comp) @ (n_comp, n_feat)
        residual = x - latent @ self.C.T  # negate so can use in-place add

        # Adaptive learning rate: (n_obs,)
        alpha = self.base_eta / (1.0 + self.base_eta * torch.norm(latent, dim=1))

        # (n_feat, n_comp) += ((n_obs, 1) * (n_obs, n_feat)).T @ (n_obs, n_comp)
        self.C.add_((alpha[:, None] * residual).T @ latent)

        # inv((n_comp, n_feat) @ (n_feat, n_comp)) @ (n_comp, n_feat)
        self.C_cross.copy_(torch.linalg.inv_ex(self.C.T @ self.C)[0] @ self.C.T)

    def get_components(self) -> Tensor:
        # useful for visualization against exact SVD on data matrix
        U, S, V = torch.svd(self.C_cross.T)
        return U.flip(1).mT

    def transform(self, x: Tensor) -> Tensor:
        # (n_obs, n_comp) = (n_obs, n_feat) @ (n_feat, n_comp)
        return x @ self.C_cross.T  # avoid transpose with batch dim

    def inverse_transform(self, z: Tensor) -> Tensor:
        # (n_obs, n_feat) = (n_obs, n_comp) @ (n_comp, n_feat)
        return z @ self.C.T  # avoid transpose with batch dim
