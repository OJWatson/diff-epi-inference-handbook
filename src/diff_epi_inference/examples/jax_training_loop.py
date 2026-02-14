"""A tiny JAX-based training loop (behind the ``modern-sbi`` extra).

This is intentionally minimal: it trains a small Equinox MLP on a synthetic
regression task using Optax.

Why keep this here?

- The handbook's M4 chapter introduces the "modern SBI" stack.
- We want one concrete, runnable example showing the mechanics of a JAX training
  loop (params, grads, opt state, jit, PRNG) without pulling in a large library.

Usage
-----

Install the extra:

    pip install ".[modern-sbi]"

Then run:

    python -m diff_epi_inference.examples.jax_training_loop

This module should import and run only when the ``modern-sbi`` dependencies
(JAX + equinox + optax) are available.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_modern_sbi() -> tuple[object, object, object]:
    """Import JAX stack lazily so core imports remain NumPy-only."""

    try:
        import equinox as eqx  # type: ignore
        import jax  # type: ignore
        import optax  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "This example requires the 'modern-sbi' extra. "
            "Install with: pip install '.[modern-sbi]'"
        ) from e
    return jax, eqx, optax


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    n_train: int = 512
    n_steps: int = 300
    batch_size: int = 64
    learning_rate: float = 3e-3
    hidden_size: int = 32


def run(cfg: TrainConfig | None = None) -> dict[str, float]:
    """Run a tiny supervised training loop.

    Returns a small dict of metrics for quick smoke-testing / demo printing.
    """

    if cfg is None:
        cfg = TrainConfig()

    jax, eqx, optax = _require_modern_sbi()
    import jax.numpy as jnp  # type: ignore

    key = jax.random.PRNGKey(cfg.seed)

    # Synthetic regression: y = 2x + 1 + eps
    key, kx, kn = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (cfg.n_train, 1), minval=-1.0, maxval=1.0)
    y = 2.0 * x + 1.0 + 0.1 * jax.random.normal(kn, (cfg.n_train, 1))

    # Small MLP.
    key, kmodel = jax.random.split(key)
    model = eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=cfg.hidden_size,
        depth=2,
        key=kmodel,
    )

    optim = optax.adam(cfg.learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(m, xb, yb):
        pred = jax.vmap(m)(xb)
        return jnp.mean((pred - yb) ** 2)

    @eqx.filter_jit
    def step(m, state, xb, yb):
        loss, grads = loss_fn(m, xb, yb)
        updates, state = optim.update(grads, state, params=m)
        m = eqx.apply_updates(m, updates)
        return m, state, loss

    # Train.
    losses = []
    for t in range(cfg.n_steps):
        key, kb = jax.random.split(key)
        idx = jax.random.randint(kb, (cfg.batch_size,), minval=0, maxval=cfg.n_train)
        xb, yb = x[idx], y[idx]
        model, opt_state, loss = step(model, opt_state, xb, yb)
        if (t % 50) == 0:
            losses.append(float(loss))

    # Evaluate on a small fixed grid.
    x_grid = jnp.linspace(-1.0, 1.0, 128)[:, None]
    y_true = 2.0 * x_grid + 1.0
    y_pred = jax.vmap(model)(x_grid)
    mse = float(jnp.mean((y_pred - y_true) ** 2))

    return {"mse_grid": mse, "loss_t0_t50_t100_...": float(jnp.array(losses).mean())}


def main() -> None:  # pragma: no cover
    metrics = run()
    print(metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
