from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(x: np.ndarray, *, as_row: bool) -> np.ndarray:
    """Ensure x is 2D.

    Parameters
    ----------
    as_row:
        If x is 1D, interpret it as a single row (shape (1, d)) when True, else
        as a column (shape (d, 1)).
    """

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[None, :] if as_row else x[:, None]
    if x.ndim != 2:
        raise ValueError("Expected a 1D or 2D array")
    return x


def _stdnorm_logpdf(z: np.ndarray) -> np.ndarray:
    """Row-wise log density of standard normal N(0, I)."""
    z = np.asarray(z, dtype=float)
    d = z.shape[1]
    return -0.5 * (np.sum(z**2, axis=1) + d * np.log(2 * np.pi))


@dataclass(frozen=True)
class ConditionalAffineDiagNormal:
    """A minimal conditional affine normalising flow (diagonal Gaussian).

    For context c and base z ~ N(0, I):

        theta = mu(c) + sigma * z

    where mu(c) is affine in c, and sigma is a positive diagonal scale.

    Notes
    -----
    - This is equivalent to a conditional diagonal Gaussian with a linear mean.
    - We keep sigma global (not context-dependent) to keep the baseline stable
      and closed-form fit. More expressive flows can come later behind optional
      JAX extras.
    """

    # Mean model: mu = b + C @ W
    b: np.ndarray  # (d,)
    W: np.ndarray  # (k, d)

    # Diagonal scale parameters.
    log_sigma: np.ndarray  # (d,)

    @property
    def dim(self) -> int:
        return int(np.asarray(self.b).shape[0])

    @property
    def context_dim(self) -> int:
        return int(np.asarray(self.W).shape[0])

    def mean(self, context: np.ndarray) -> np.ndarray:
        C = _as_2d(context, as_row=True)
        return C @ self.W + self.b[None, :]

    def sample(self, context: np.ndarray, n: int, *, rng: np.random.Generator) -> np.ndarray:
        C = _as_2d(context, as_row=True)
        if C.shape[0] != 1:
            raise ValueError("context must be a single context row (shape (k,) or (1,k))")
        mu = self.mean(C)[0]
        sigma = np.exp(self.log_sigma)
        z = rng.normal(size=(int(n), self.dim))
        return mu[None, :] + z * sigma[None, :]

    def log_prob(self, theta: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Return log q(theta | context) for each row of theta.

        Parameters
        ----------
        theta:
            Array of shape (n, d) or (d,).
        context:
            Context array of shape (n, k), (k,), or (1, k). If a single context is
            provided, it is broadcast across theta rows.
        """

        X = _as_2d(theta, as_row=True)
        C = _as_2d(context, as_row=True)
        if C.shape[0] == 1 and X.shape[0] > 1:
            C = np.repeat(C, repeats=X.shape[0], axis=0)
        if C.shape[0] != X.shape[0]:
            raise ValueError("context rows must match theta rows (or be a single row)")

        mu = self.mean(C)
        sigma = np.exp(self.log_sigma)[None, :]
        z = (X - mu) / sigma
        log_det_inv = -np.sum(self.log_sigma)  # constant for diagonal affine
        return _stdnorm_logpdf(z) + log_det_inv

    @staticmethod
    def fit_closed_form(
        *,
        contexts: np.ndarray,
        thetas: np.ndarray,
        ridge: float = 1e-8,
        min_sigma: float = 1e-6,
    ) -> "ConditionalAffineDiagNormal":
        """Fit a conditional diagonal Gaussian with linear mean.

        We solve least squares for the mean parameters:

            theta ≈ b + C @ W

        and then set sigma to the per-dimension residual std.

        Parameters
        ----------
        contexts:
            (n, k) context matrix.
        thetas:
            (n, d) parameter matrix.
        ridge:
            Small L2 regularisation for numerical stability.
        min_sigma:
            Lower bound for sigma to avoid degenerate densities.
        """

        C = _as_2d(contexts, as_row=True)
        X = _as_2d(thetas, as_row=True)
        if C.shape[0] != X.shape[0]:
            raise ValueError("contexts and thetas must have the same n")

        n, k = C.shape
        d = X.shape[1]

        # Add intercept: [1, C]
        A = np.column_stack([np.ones(n), C])

        # Ridge-regularised least squares: (A^T A + λI)^{-1} A^T X
        ATA = A.T @ A
        ATA = ATA + ridge * np.eye(ATA.shape[0])
        coef = np.linalg.solve(ATA, A.T @ X)  # (k+1, d)

        b = coef[0]
        W = coef[1:]

        resid = X - (A @ coef)
        sigma = np.std(resid, axis=0, ddof=0)
        sigma = np.maximum(sigma, float(min_sigma))
        log_sigma = np.log(sigma)

        return ConditionalAffineDiagNormal(b=b, W=W, log_sigma=log_sigma)
