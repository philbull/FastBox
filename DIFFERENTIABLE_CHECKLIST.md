# Differentiability branch: checklist

Working notes for the `differentiable` branch. Delete this file before
merging into `main`.

## Status

- [x] Phil's `FastBox-JAX.patch` applied unchanged (76db298), on top of
      `main` at e8fd1fd.
- [x] JAX tests fixed for current JAX (8872388).
      `jax.experimental.enable_x64` no longer exists, so all three tests
      failed on JAX 0.12.1. The full suite now passes (43 tests) on JAX
      0.12.1, and the JAX tests also pass on 0.4.35.
- [x] The README `jax.jit(jax.grad(...))` example runs and returns finite
      gradients.

## Decisions

- [ ] **Separate module or a single code path?** The patch adds
      `fastbox/jax_backend.py` alongside the NumPy implementation, so
      density, RSD, beam and power now each exist twice. Parity tests
      catch drift, but every future change has to be made in both places.
      The alternative is to make the existing `CosmoBox`/`Power` methods
      work with JAX arrays directly (e.g. an `xp` array-namespace switch).
      That is more invasive but leaves one code path.
- [ ] **What is the first target?** The patch targets field-level
      inference: a fixed realisation goes through density, RSD, beam and
      power. If the near-term use is gradients of *predicted* spectra for
      forecasts and Fisher matrices, port `Power.predict_hi_power_spectrum`,
      `predict_galaxy_power_spectrum` and `predict_cross_spectrum` instead
      (they already take a tabulated P(k)).
- [ ] **Should the NumPy density sampler match?** `jax_backend.gaussian_density`
      filters real white noise. `CosmoBox.realise_density` keeps the real
      part of the inverse FFT of independent complex draws, then stores
      `delta_k = fft(delta_x)`. Both give the same statistics (checked by
      `test_gaussian_density_recovers_input_power`), but the same seed
      gives different fields in the two paths.

      Changing the NumPy sampler would change every existing seeded
      realisation, including the paper figures
      (`scripts/make_paper_figures.py` uses `np.random.seed(41)`).
      Suggestion: leave it unchanged. If NumPy-JAX field matching is ever
      needed, add an optional `white_noise=` argument to `realise_density`
      that leaves the default unchanged.
- [ ] **Cosmology gradients.** pyCCL is not differentiable. Options:
      leave it out of scope (fixed cosmology, as now), use jax-cosmo, or
      use an emulator. The kernels already take P(k), f, aH as inputs, so
      this can wait.
- [ ] **Minimum JAX version.** `setup.py` says `jax>=0.4.35`. Keep it
      (tests now handle both APIs) or raise it and drop the fallback?
- [ ] **Behaviour for negative `sigma_v`.** `jnp.maximum(sigma_v, 0)`
      matches the NumPy code (negative means no damping), but its gradient
      is zero for `sigma_v < 0`. An optimiser that steps below zero gets
      stuck there. Options: keep as is, reparametrise (e.g. fit
      `log sigma_v`), or raise.
- [ ] **Land the two refactors separately?** `CosmoBox.matter_power_grid`
      and `Power._power_bin_edges` don't need JAX and leave existing results
      unchanged (the 40 existing tests pass). They could go to `main` on
      their own now.

## To do before merging

- [x] **CI.** `unit-tests.yaml` installs `.` without JAX, and
      `test_jax_backend.py` skips itself when JAX is missing. CI stays green
      while never running the JAX tests. Install `.[jax]` in at least one
      matrix job. *Done: the Python 3.13 job installs `.[jax]` and fails if
      JAX is missing; the 3.10 job still runs without JAX.*
- [ ] **Docs.** Add `docs/jax_backend.md` to match the other modules, and
      check that the strict Read the Docs build works. JAX may need adding
      to `docs/requirements.txt` or mocking.
- [ ] **README example.** The current statistic (`nansum` of the power) is
      a toy. Replace it with a small Gaussian likelihood against a fixed
      data vector so it shows the intended use.
- [ ] **Precision.** The README example runs in float32 by default and the
      tests use float64. State which one users should use for inference.
- [ ] **Review `jax_backend.py` by hand.** The patch is machine-generated.
      Parity and finite-difference tests pass, but the code still needs
      the normal review. One kernel at a time:
  - [x] `gaussian_density`: normalisation and statistics correct. Added a
        test that the mean power over 20 realisations matches the input
        P(k) in every shell (for both samplers), and an exact gradient test
        (d var/dA = var when P is scaled by A). Docstring now states the
        P(k) symmetry requirement and the difference from `realise_density`.
  - [ ] `linear_rsd_density`
  - [ ] `convolve_gaussian_beam`
  - [ ] `estimate_power_spectrum`
- [ ] **JOSS paper.** Decide whether the paper's Software Design section
      mentions the JAX backend, or leave it out until it is merged.

## Out of scope for now (needs a modelling decision, not just a port)

- Galaxy/halo catalogues: Poisson sampling has no pathwise gradient, and
  the output arrays vary in length. For inference, use the expected-count
  field or a Poisson likelihood.
- Interpolated RSD (`scipy.interpolate.griddata` on displaced coordinates):
  needs a JAX remapping scheme with explicit periodicity.
- Foreground cleaning/PCA: a fixed projector is easy. A data-dependent
  fit needs care where retained and discarded eigenvalues are close.
