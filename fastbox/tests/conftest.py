"""Shared fixtures and helpers for the FastBox tests.

Most tests compare FastBox against an independent reference (an analytic
result, a brute-force sum, or the exact kernel on the grid) rather than
against FastBox itself. Grids are kept small so the suite runs quickly.
"""
import numpy as np
import pyccl as ccl
import pytest

from fastbox.box import CosmoBox, default_cosmo
from fastbox.power import Power


@pytest.fixture(autouse=True)
def seed():
    """FastBox draws from the global NumPy generator, so seed it per test."""
    np.random.seed(1234)


@pytest.fixture
def make_box():
    """Factory for boxes with the default cosmology and no realisation."""
    def _make(box_scale=1000., nsamp=32, redshift=0.8, **kwargs):
        return CosmoBox(cosmo=default_cosmo, box_scale=box_scale, nsamp=nsamp,
                        redshift=redshift, realise_now=False, **kwargs)
    return _make


@pytest.fixture
def theory():
    """Wide-range k grid and power spectrum to pass to the forward models."""
    def _theory(box, linear=False):
        km = np.logspace(-4, 1, 2000)
        fn = ccl.linear_matter_power if linear else ccl.nonlin_matter_power
        return km, fn(box.cosmo, km, box.scale_factor)
    return _theory


def kmag(box):
    """|k| on the FFT grid, computed the same way as the forward models.

    This can differ from box.k in the last bit, which moves modes that sit
    exactly on a bin edge, so use it when comparing against a model exactly.
    """
    kx, ky, kz = [2. * np.pi * np.fft.fftfreq(n, d=L / n) for n, L in
                  [(box.Nx, box.Lx), (box.Ny, box.Ly), (box.Nz, box.Lz)]]
    return np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2
                   + kz[None, None, :]**2)


def shell_bin(box, arr3d):
    """Average a 3D Fourier-space array in the k-shells used by Power."""
    kf, kn = Power(box)._isotropic_k_range()
    bins = np.arange(kf, kn + kf, kf)
    idx = np.digitize(kmag(box).flatten(), bins)
    a = np.broadcast_to(arr3d, box.k.shape).flatten()
    return np.array([a[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(1, bins.size)])


def cic_kernel(box):
    """Exact CIC window amplitude for galaxies scattered uniformly in a voxel.

    Re-assigning such a galaxy with CIC gives a discrete [1/8, 3/4, 1/8]
    kernel per axis, whose Fourier amplitude is 3/4 + cos(k dx)/4.
    """
    out = np.ones(box.k.shape)
    for axis, (n, L) in enumerate([(box.Nx, box.Lx), (box.Ny, box.Ly),
                                   (box.Nz, box.Lz)]):
        dx = L / n
        k = 2. * np.pi * np.fft.fftfreq(n, dx)
        shape = [1, 1, 1]
        shape[axis] = n
        out = out * (0.75 + 0.25 * np.cos(k * dx)).reshape(shape)
    return out
