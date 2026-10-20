"""Tests for the MeerKLASS helpers, filters and foregrounds on non-square grids."""
import numpy as np
import pytest

import fastbox.meerklass as mk
from fastbox.filters import angular_bandpass_filter
from fastbox.foregrounds import ForegroundModel


@pytest.fixture
def survey(make_box):
    """A small non-cubic box with an elliptical footprint and tapered hits."""
    nx, ny, nz = 30, 20, 16
    freqs = np.linspace(971., 1023., nz)
    box = make_box(box_scale=(600., 400., 300.), nsamp=(nx, ny, nz),
                   redshift=0.39, frequency_array=freqs[::-1])
    X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny),
                       indexing='ij')
    rr = X**2 + (Y / 0.8)**2
    mask = rr < 0.8
    hits = (np.clip(1. - rr, 0., None) * 400.)[..., None] * np.ones(nz)
    return box, mask, hits


def test_hitmap_noise_seeding(survey):
    box, mask, hits = survey
    freqs = np.sort(box.freq_array())
    n1 = mk.hitmap_thermal_noise(box, hits, mask, freqs)
    n2 = mk.hitmap_thermal_noise(box, hits, mask, freqs)
    assert not np.array_equal(n1, n2)
    s1 = mk.hitmap_thermal_noise(box, hits, mask, freqs, seed=7)
    s2 = mk.hitmap_thermal_noise(box, hits, mask, freqs, seed=7)
    assert np.array_equal(s1, s2)


def test_hitmap_noise_descending_frequencies(survey):
    """box.freq_array() is descending; that must not give NaNs."""
    box, mask, hits = survey
    noise = mk.hitmap_thermal_noise(box, hits, mask, box.freq_array())
    assert np.all(np.isfinite(noise))
    assert np.all(noise[~mask] == 0.)


def test_hitmap_noise_radiometer_level(survey):
    """Noise level follows T_sys / sqrt(2 dnu t_p) with the MeerKLASS T_sys."""
    box, mask, hits = survey
    freqs = np.sort(box.freq_array())
    t_sample = 2.
    noise = mk.hitmap_thermal_noise(box, hits, mask, freqs, t_sample_s=t_sample,
                                    seed=1)
    valid = mask[..., None] & (hits > 0)
    # Normalise out the hit count, then compare the per-channel scatter
    z = noise * np.sqrt(np.where(valid, hits * t_sample, 1.))
    dnu = abs(np.mean(np.diff(freqs))) * 1e6
    nu = freqs / 1e3
    Tsys = (7.5 + 10. * (nu - 0.75)**2 + 3. + 2.725
            + 25. * (408. / freqs)**2.75) * 1e3
    pred = Tsys / np.sqrt(2. * dnu)
    meas = np.array([z[..., i][valid[..., i]].std()
                     for i in range(freqs.size)])
    assert np.allclose(meas, pred, rtol=0.1)
    assert abs(np.median(meas / pred) - 1.) < 0.02


def test_pca_masked_weighted(survey):
    """Removing 3 modes from rank-3 foregrounds leaves only the projected signal."""
    box, mask, hits = survey
    nx, ny, nz = hits.shape
    nu = np.sort(box.freq_array()) / 1000.
    fg = sum(np.random.lognormal(3, 0.3, (nx, ny))[..., None] * 1e3
             * (nu**b)[None, None, :] for b in [-2.7, -2.5, -2.9])
    sig = np.random.normal(0., 0.1, (nx, ny, nz))
    cube = fg + sig

    cl0 = mk.pca_masked_weighted(0, cube, mask, hits)
    assert np.allclose(cl0[mask], cube[mask]) and np.all(cl0[~mask] == 0.)

    cl, A, *_ = mk.pca_masked_weighted(3, cube, mask, hits, return_modes=True)
    residual = cl[mask] - sig[mask]
    projected = sig[mask] @ A @ A.T
    assert np.allclose(residual, -projected, atol=1e-6 * np.abs(fg).max())


def test_angular_bandpass_filter_nonsquare():
    """A plane wave inside the band survives; one outside it is removed."""
    nx, ny, nz = 60, 36, 4
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    inside = np.cos(2. * np.pi * 6. * x / nx) * np.ones((nx, ny, nz))   # 0.1
    outside = np.cos(2. * np.pi * 12. * y / ny) * np.ones((nx, ny, nz))  # 0.33
    out = angular_bandpass_filter(inside + outside, 0.05, 0.2).real
    assert np.allclose(out, inside, atol=1e-10)


def test_foreground_model_noncubic(survey):
    box, _, _ = survey
    amp = ForegroundModel(box).realise_foreground_amp(700., -2.4, 30e3,
                                                      smoothing_scale=0.5)
    assert amp.shape == (box.Nx, box.Ny)
    assert np.all(np.isfinite(amp))
