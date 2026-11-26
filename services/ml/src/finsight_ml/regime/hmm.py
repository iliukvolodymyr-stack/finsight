from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import logsumexp

RegimeLabel = Literal["bull", "bear", "range"]


@dataclass(frozen=True)
class RegimeOutput:
    regime: list[str]
    proba: np.ndarray  # columns match ["bear", "range", "bull"]


def build_regime_observations(close: pd.Series, vol_span: int = 24) -> pd.DataFrame:
    """
    Build strictly causal 2D stationary observation matrix `[logret_1, ewm_vol]` for HMM.
    """
    c = close.astype("float64")
    log_ret = np.log(c / c.shift(1))
    vol = log_ret.ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    return pd.DataFrame({"logret_1": log_ret, "ewm_vol": vol}, index=close.index)


class CausalGaussianHMM:
    """
    Strictly causal 3-state Diagonal Gaussian HMM baseline (`bear`, `range`, `bull`).

    Training / Inference separation (docs/04-ml-pipeline.md §5):
    - `fit()` estimates transition matrix `A`, initial distribution `pi`, and
      per-state Gaussian means/variances via log-space Baum-Welch EM on the
      training slice ONLY.
    - States are deterministically ordered by unstandardized return mean:
      `mu_return(bear) <= mu_return(range) <= mu_return(bull)` to avoid label switching.
    - `filter_causal()` runs ONLY the forward filter `alpha_t = P(s_t | x_{1:t})`
      with zero backward smoothing, guaranteeing zero future look-ahead.
    """

    STATE_NAMES: tuple[str, str, str] = ("bear", "range", "bull")

    def __init__(self, n_iter: int = 35, tol: float = 1e-4, min_var: float = 1e-3) -> None:
        self.n_states = 3
        self.n_iter = n_iter
        self.tol = tol
        self.min_var = min_var

        self.feat_mean: np.ndarray = np.zeros(2, dtype=np.float64)
        self.feat_std: np.ndarray = np.ones(2, dtype=np.float64)
        self.pi: np.ndarray = np.full(3, 1.0 / 3.0, dtype=np.float64)
        self.transmat: np.ndarray = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)
        self.means: np.ndarray = np.zeros((3, 2), dtype=np.float64)
        self.vars: np.ndarray = np.ones((3, 2), dtype=np.float64)
        self.is_fitted: bool = False

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return np.asarray((X - self.feat_mean) / np.maximum(self.feat_std, 1e-8), dtype=np.float64)

    def _log_emission(self, X_std: np.ndarray) -> np.ndarray:
        # X_std: (T, D), means: (K, D), vars: (K, D) -> (T, K)
        diff = X_std[:, None, :] - self.means[None, :, :]
        inv_var = 1.0 / np.maximum(self.vars[None, :, :], self.min_var)
        log_det = np.sum(np.log(2.0 * np.pi * np.maximum(self.vars, self.min_var)), axis=1)
        quad = np.sum((diff**2) * inv_var, axis=2)
        return np.asarray(-0.5 * (log_det[None, :] + quad), dtype=np.float64)

    def _forward_log(
        self, log_B: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
    ) -> tuple[np.ndarray, float]:
        t_len, k_len = log_B.shape
        log_alpha = np.empty((t_len, k_len), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, t_len):
            prev = log_alpha[t - 1, :, None] + log_A
            m = prev.max(axis=0)
            log_alpha[t] = m + np.log(np.exp(prev - m).sum(axis=0)) + log_B[t]
        last = log_alpha[-1]
        m_last = float(last.max())
        ll = m_last + float(np.log(np.exp(last - m_last).sum()))
        return log_alpha, ll

    def _backward_log(self, log_B: np.ndarray, log_A: np.ndarray) -> np.ndarray:
        t_len, k_len = log_B.shape
        log_beta = np.zeros((t_len, k_len), dtype=np.float64)
        for t in range(t_len - 2, -1, -1):
            nxt = log_A + (log_B[t + 1] + log_beta[t + 1])[None, :]
            m = nxt.max(axis=1)
            log_beta[t] = m + np.log(np.exp(nxt - m[:, None]).sum(axis=1))
        return log_beta

    def fit(self, obs: pd.DataFrame | np.ndarray) -> "CausalGaussianHMM":
        arr = (
            obs.dropna().to_numpy(dtype=np.float64)
            if isinstance(obs, pd.DataFrame)
            else np.asarray(obs, dtype=np.float64)
        )
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if len(arr) < 30:
            raise ValueError("Need at least 30 finite observations to fit CausalGaussianHMM.")

        self.feat_mean = np.mean(arr, axis=0)
        self.feat_std = np.maximum(np.std(arr, axis=0), 1e-8)
        X_std = self._standardize(arr)

        # Deterministic quantile initialization on return dimension
        ret_col = X_std[:, 0]
        q_lo, q_hi = np.quantile(ret_col, [0.33, 0.67])
        masks = [ret_col <= q_lo, (ret_col > q_lo) & (ret_col < q_hi), ret_col >= q_hi]
        self.means = np.zeros((3, 2), dtype=np.float64)
        self.vars = np.ones((3, 2), dtype=np.float64)
        for k, m in enumerate(masks):
            if np.sum(m) >= 3:
                self.means[k] = np.mean(X_std[m], axis=0)
                self.vars[k] = np.maximum(np.var(X_std[m], axis=0), self.min_var)
            else:
                self.means[k] = np.array([(k - 1) * 0.5, 0.0], dtype=np.float64)

        self.pi = np.full(3, 1.0 / 3.0, dtype=np.float64)
        # Sticky prior encourages realistic persistent regimes
        self.transmat = np.array(
            [
                [0.90, 0.08, 0.02],
                [0.05, 0.90, 0.05],
                [0.02, 0.08, 0.90],
            ],
            dtype=np.float64,
        )

        prev_ll = -np.inf
        t_len = len(X_std)

        for _ in range(self.n_iter):
            log_B = self._log_emission(X_std)
            log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))
            log_A = np.log(np.clip(self.transmat, 1e-12, 1.0))

            log_alpha, ll = self._forward_log(log_B, log_pi, log_A)
            log_beta = self._backward_log(log_B, log_A)

            # Posterior gamma_t(k) = P(s_t = k | X_{1:T}) during EM training
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Expected transition counts xi_{t, i, j} (vectorized over t, zero Python loop)
            if t_len > 1:
                log_xi = (
                    log_alpha[:-1, :, None]
                    + log_A[None, :, :]
                    + (log_B[1:, None, :] + log_beta[1:, None, :])
                )
                log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
                xi_sum = np.sum(np.exp(log_xi), axis=0)
            else:
                xi_sum = np.zeros((3, 3), dtype=np.float64)

            # M-step
            self.pi = np.maximum(gamma[0], 1e-6)
            self.pi /= np.sum(self.pi)

            row_sums = np.sum(xi_sum, axis=1, keepdims=True)
            self.transmat = np.where(
                row_sums > 1e-12, xi_sum / np.maximum(row_sums, 1e-12), 1.0 / 3.0
            )
            self.transmat = np.maximum(self.transmat, 1e-6)
            self.transmat /= np.sum(self.transmat, axis=1, keepdims=True)

            gamma_sum = np.maximum(np.sum(gamma, axis=0), 1e-6)
            self.means = (gamma.T @ X_std) / gamma_sum[:, None]
            for k in range(3):
                diff = X_std - self.means[k]
                self.vars[k] = np.maximum(
                    (gamma[:, k] @ (diff**2)) / gamma_sum[k],
                    self.min_var,
                )

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Canonical ordering by return mean (bear < range < bull) to eliminate label switching
        order = np.argsort(self.means[:, 0])
        self.pi = self.pi[order]
        self.transmat = self.transmat[np.ix_(order, order)]
        self.means = self.means[order]
        self.vars = self.vars[order]
        self.is_fitted = True
        return self

    def filter_causal(self, obs: pd.DataFrame | np.ndarray) -> RegimeOutput:
        """
        Compute strictly causal filtered state probabilities `P(s_t | x_{1:t})`.
        Rows with NaN inputs hold the prior propagated state transition cleanly.
        """
        if not self.is_fitted:
            raise RuntimeError("CausalGaussianHMM must be fitted before calling filter_causal.")

        arr = (
            obs.to_numpy(dtype=np.float64)
            if isinstance(obs, pd.DataFrame)
            else np.asarray(obs, dtype=np.float64)
        )
        n = len(arr)
        probs = np.zeros((n, 3), dtype=np.float64)
        labels: list[str] = ["range"] * n
        finite_mask = np.all(np.isfinite(arr), axis=1)

        # Precompute scaled emission likelihoods exp(log_B_t - max_k log_B_{t,k}) once in batch;
        # row-wise constant scaling cancels out identically in posterior normalization.
        b_scaled = np.ones((n, 3), dtype=np.float64)
        if np.any(finite_mask):
            x_std_all = self._standardize(arr[finite_mask])
            log_b_all = self._log_emission(x_std_all)
            log_b_all -= np.max(log_b_all, axis=1, keepdims=True)
            b_scaled[finite_mask] = np.exp(log_b_all)

        state_prob = self.pi.copy()
        transmat = self.transmat
        state_names = self.STATE_NAMES
        for t in range(n):
            prior = self.pi if t == 0 else (state_prob @ transmat)
            if finite_mask[t]:
                unnorm = np.clip(prior, 1e-12, 1.0) * b_scaled[t]
                denom = float(unnorm.sum())
                state_prob = unnorm / denom if denom > 0.0 else prior
            else:
                denom = float(prior.sum())
                state_prob = prior / denom if denom > 0.0 else self.pi.copy()
            probs[t] = state_prob
            labels[t] = state_names[int(np.argmax(state_prob))]

        return RegimeOutput(regime=labels, proba=probs)
