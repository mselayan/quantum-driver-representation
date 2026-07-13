"""
The quantum-inspired driver model.

Learnable parameters:
  V_k   (K matrices, D x rank)  profile factors; rho_k = V_k V_k^T / tr(.)
  beta  (K x q)                 context-to-profile activation weights

Fixed hyperparameters:
  alpha  state-evolution blend. A learnable alpha collapses toward zero, which
         switches off the context-driven mixture and leaves only the
         observation update.
  eta    behavioral adaptation strength.

eta is fixed by construction, not merely by choice. It appears only in rho_next,
which is detached into the per-driver state before the next frame. The loss is a
function of rho_t, which does not depend on eta, so no gradient can reach it. It
is carried as a float rather than a parameter so that the code says what it is.

Density-matrix constraints (symmetric, PSD, unit trace) hold by construction
through the V_k V_k^T / tr(V_k V_k^T) parameterization.
"""

import numpy as np
import torch
import torch.nn as nn


class QuantumDriverModel(nn.Module):

    def __init__(self, K, D, q, rank=10, alpha=0.2, eta=0.1):
        super().__init__()
        self.K = K
        self.D = D
        self.rank = rank
        self.alpha = alpha
        self.eta = eta

        self.Vs = nn.ParameterList([
            nn.Parameter(torch.randn(D, rank) * 0.1) for _ in range(K)
        ])
        self.beta = nn.Parameter(torch.randn(K, q) * 0.05)

    def get_alpha(self):
        return self.alpha

    def get_eta(self):
        return self.eta

    def build_profiles(self):
        """rho_k = V_k V_k^T / tr(V_k V_k^T), stacked as (K, D, D)."""
        profiles = []
        for V in self.Vs:
            M = V @ V.T
            tr = torch.trace(M)
            if tr < 1e-12:
                profiles.append(torch.eye(self.D, device=V.device) / self.D)
            else:
                profiles.append(M / tr)
        return torch.stack(profiles)

    def softmax_activation(self, c):
        """pi_k(c) = softmax(beta c). c is (q,) -> returns (K,)."""
        return torch.softmax(self.beta @ c, dim=0)

    def forward_chunk(self, Phi_chunk, C_chunk, ids_chunk, driver_states):
        """Negative log-likelihood over one contiguous chunk of frames.

        Per frame:
          pi       = softmax(beta c_t)                context activation
          rho_t    = (1-alpha) rho_prev + alpha sum_k pi_k rho_k   state evolution
          p        = phi_t^T rho_t phi_t              Born-rule likelihood
          rho_next = (1-eta) rho_t + eta phi_t phi_t^T   behavioral adaptation

        driver_states carries rho_prev across frames, detached. The detach means
        rho_next is a value, not a node in the graph, so alpha, beta and V are
        fit against a single frame's likelihood at a time, and eta receives no
        gradient at all. eta is therefore a fixed hyperparameter.
        """
        device = Phi_chunk.device
        alpha = self.get_alpha()
        eta = self.get_eta()
        rho_k = self.build_profiles()

        identity = torch.eye(self.D, device=device) / self.D
        nll = torch.zeros((), device=device)

        for i in range(Phi_chunk.shape[0]):
            driver = ids_chunk[i]
            phi = Phi_chunk[i]
            c = C_chunk[i]

            if driver not in driver_states:
                driver_states[driver] = identity.clone()
            rho_prev = driver_states[driver]

            pi = self.softmax_activation(c)
            mixture = torch.einsum("k,kde->de", pi, rho_k)
            rho_t = (1.0 - alpha) * rho_prev + alpha * mixture

            p = torch.clamp(phi @ rho_t @ phi, min=1e-12)
            nll = nll - torch.log(p)

            rho_next = (1.0 - eta) * rho_t + eta * torch.outer(phi, phi)
            driver_states[driver] = rho_next.detach()

        return nll, driver_states


def von_neumann_entropy(rho_k):
    """Sum of S(rho_k) = -tr(rho_k log rho_k) over the K profiles, in nats.

    Used as the gamma-weighted regularizer. Subtracting gamma * S from the NLL
    rewards spectrally spread profiles, so a profile can occupy several
    eigendirections instead of collapsing to rank one.
    """
    total = torch.zeros((), device=rho_k.device)
    for k in range(rho_k.shape[0]):
        eigvals = torch.clamp(torch.linalg.eigvalsh(rho_k[k]), min=1e-12)
        total = total - (eigvals * torch.log(eigvals)).sum()
    return total


def enforce_density_matrix(rho):
    """Symmetrize, clip negative eigenvalues, renormalize the trace. NumPy."""
    rho = (rho + rho.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total < 1e-12:
        return np.eye(rho.shape[0]) / rho.shape[0]
    eigvals /= total
    return (eigvecs * eigvals) @ eigvecs.T


def top_eigenmodes(rho, n_modes):
    """Top-n eigenvalues and eigenvectors of rho, descending. NumPy."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order][:n_modes], eigvecs[:, order][:, :n_modes]
