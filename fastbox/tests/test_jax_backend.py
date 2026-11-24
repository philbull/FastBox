import numpy as np
import pytest

from fastbox.beams import GaussianBeamModel
from fastbox.power import Power

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from fastbox import jax_backend as fbj


def test_gaussian_density_normalisation_and_zero_mode(make_box):
    box = make_box(box_scale=(305., 470., 640.), nsamp=(9, 10, 11))
    pk = box.matter_power_grid(linear=True)
    with jax.experimental.enable_x64():
        white_noise = jax.random.normal(jax.random.key(0), pk.shape)
        density = jax.jit(fbj.gaussian_density)(white_noise, pk, box.boxfactor)
        # The legacy sampler keeps the real part of the inverse FFT of
        # independent complex draws. Construct its Hermitian input directly
        # from the same voxel noise to compare the two normalisations.
        white_modes = np.fft.fftn(white_noise) / np.sqrt(pk.size)
        expected = np.fft.ifftn(white_modes * np.sqrt(pk * box.boxfactor)).real
        np.testing.assert_allclose(density, expected, atol=1e-13)
        assert abs(np.mean(density)) < 1e-13

        def variance(spectrum):
            field = fbj.gaussian_density(white_noise, spectrum, box.boxfactor)
            return jnp.mean(field**2)

        derivative = jax.jit(jax.grad(variance))(pk)
        assert derivative[0, 0, 0] == 0.0
        assert np.all(np.isfinite(derivative))


def test_rsd_beam_power_parity_and_gradients(make_box):
    box = make_box(box_scale=(305., 470., 640.), nsamp=(9, 10, 11))
    field = np.random.default_rng(0).normal(size=box.k.shape)
    ang_x, ang_y = box.pixel_array()
    rsd = fbj.prepare_rsd(box)
    bins = fbj.prepare_power_bins(box)
    power = Power(box)

    def forward(bias, sigma_v, width):
        shifted = fbj.linear_rsd_density(field, rsd, bias=bias, sigma_v=sigma_v)
        smoothed = fbj.convolve_gaussian_beam(shifted, ang_x, ang_y, width)
        return fbj.estimate_power_spectrum(smoothed, bins, box.boxfactor)

    def reference(bias, sigma_v, width):
        shifted = box.linear_rsd_density(field, bias=bias, sigma_v=sigma_v)
        smoothed = GaussianBeamModel(box, width).convolve_fft(shifted)
        return power.estimate_power_spectrum(smoothed)

    def loss(params):
        return jnp.nansum(forward(*params)[1])

    params = np.array([1.6, 115.0, 0.8])
    with jax.experimental.enable_x64():
        for got, expected in zip(jax.jit(forward)(*params), reference(*params)):
            np.testing.assert_allclose(got, expected, rtol=1e-12)
        derivative = jax.jit(jax.grad(loss))(params)
        steps = np.diag([1e-5, 1e-3, 1e-5])
        finite_difference = [
            (np.nansum(reference(*(params + step))[1])
             - np.nansum(reference(*(params - step))[1])) / (2 * np.linalg.norm(step))
            for step in steps
        ]
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-6)
        zero_dispersion = jax.grad(lambda sigma: loss(jnp.array([1.6, sigma, 0.8])))
        assert zero_dispersion(0.0) == 0.0


def test_weighted_cross_power_and_empty_shells(make_box, monkeypatch):
    box = make_box(box_scale=(2000., 400., 400.), nsamp=(5, 10, 11))
    # Insert a shell strictly between the first two nonzero x-axis modes.
    edges = box.kmin * np.array([0.5, 1.1, 1.5, 2.1])
    monkeypatch.setattr(Power, "_power_bin_edges", lambda self: edges)
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(2, *box.k.shape))
    w1, w2 = rng.uniform(0.1, 1.0, size=(2, *box.k.shape))
    bins = fbj.prepare_power_bins(box)
    expected = Power(box).estimate_weighted_power_spectrum(u, v, w1, w2)
    assert np.any(np.isnan(expected[1]))
    with jax.experimental.enable_x64():
        estimate = jax.jit(fbj.estimate_power_spectrum)
        got = estimate(u, bins, box.boxfactor, v, w1, w2)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

        def loss(field):
            _, power, errors = estimate(field, bins, box.boxfactor, v, w1, w2)
            return jnp.nansum(power) + jnp.nansum(errors)

        assert np.all(np.isfinite(jax.grad(loss)(np.zeros_like(u))))
