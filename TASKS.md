# Optimizer assignments

| Owner | Methods |
| --- | --- |
| Maya | Mini-batch stochastic gradient descent (SGD) and sharpness-aware minimization (SAM) |
| Layla | Sharpness-aware learning rate (SALR) and block coordinate descent (BCD) |

Implement update rules directly with NumPy arrays. Implement the shared
`Optimizer.train` interface, use `CountedObjective` for every gradient call,
and yield a `TrainingState` after each accepted update or block solve. Use
`shuffled_batches` for mini-batch methods and the experiment runner for
consistent initialization and checkpoint selection. Record SALR's applied
step size under the `learning_rate` diagnostics key.

- SGD takes one gradient step on each shuffled mini-batch.
- SAM computes a gradient on the mini-batch, uses it to form the normalized
  perturbation, then updates the original parameters with the gradient
  evaluated at the perturbed parameters on the same mini-batch.
- SALR uses local ascent and descent probes to estimate sharpness and adjusts
  the base learning rate using the paper's relative-sharpness schedule.
- BCD updates one parameter block at a time with the other blocks fixed.

## References

- Foret et al., [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412).
- Yue, Nouiehed, and Al Kontar, [SALR: Sharpness-Aware Learning Rate Scheduler for Improved Generalization](https://arxiv.org/abs/2011.05348).
- [SAM authors' reference implementation](https://github.com/google-research/sam).
