"""Tests for tracers, redshift-space distortions and halo catalogues."""
import numpy as np
import numpy.fft as fft
import pyccl as ccl
import pytest

import fastbox
from fastbox.halos import HaloDistribution
from fastbox.power import Power
from fastbox.tracers import GalaxyTracer
from fastbox.tests.conftest import cic_kernel, shell_bin


def smoothed_field(box, R=8.):
    """A realisation smoothed so that 1 + delta stays positive."""
    d = box.realise_density()
    return fft.ifftn(fft.fftn(d) * np.exp(-0.5 * (box.k * R)**2)).real


def test_linear_rsd_density_per_mode(make_box):
    """linear_rsd_density multiplies every mode by exactly b + f mu^2."""
    box = make_box(box_scale=2000., nsamp=32)
    d = box.realise_density(linear=True)
    b = 1.3
    ds = box.linear_rsd_density(d, bias=b, sigma_v=0.)
    f = ccl.growth_rate(box.cosmo, box.scale_factor)
    kpar = 2. * np.pi * box.Kz / box.Lz
    mu2 = np.divide(kpar**2, box.k**2, out=np.zeros_like(box.k), where=box.k > 0)
    ratio = np.abs(fft.fftn(ds))**2 / np.abs(fft.fftn(d))**2
    # Skip k = 0 and the Nyquist planes, where the mode has no partner
    half = box.Nx // 2
    good = (box.k > 0) & (np.abs(box.Kx) < half) & (np.abs(box.Ky) < half) \
        & (np.abs(box.Kz) < half)
    assert np.allclose(ratio[good], (b + f * mu2[good])**2, rtol=1e-8)


def test_apply_rsd_kaiser_boost(make_box):
    """Displacing galaxies gives the Kaiser boost of growth rate f, not a*f."""
    box = make_box(box_scale=2000., nsamp=48)
    P = Power(box)
    nbar = 2e-4
    gal = GalaxyTracer(box, nbar)
    ratio_cat, ratio_fourier = [], []
    for _ in range(3):
        d = box.realise_density(linear=True)
        pos = gal.generate_catalogue(d)
        mr = gal._assign_ngp(pos, overdensity=True)
        ms = gal._assign_ngp(gal.apply_rsd(pos, sigma_nl=0.), overdensity=True)
        k, pr, _ = P.estimate_power_spectrum(mr)
        _, ps, _ = P.estimate_power_spectrum(ms)
        ratio_cat.append((ps - 1. / nbar) / (pr - 1. / nbar))
        _, pfr, _ = P.estimate_power_spectrum(d)
        _, pfs, _ = P.estimate_power_spectrum(box.linear_rsd_density(d, sigma_v=0.))
        ratio_fourier.append(pfs / pfr)
    lo = k < 0.04
    cat = np.nanmean(np.mean(ratio_cat, 0)[lo])
    fourier = np.nanmean(np.mean(ratio_fourier, 0)[lo])
    f = ccl.growth_rate(box.cosmo, box.scale_factor)
    a = box.scale_factor
    kaiser = lambda g: 1. + 2. * g / 3. + g**2 / 5.  # monopole boost
    assert abs(fourier - kaiser(f)) < 0.05
    # The catalogue must be much closer to f than to the old a*f result. It
    # sits a few percent low because each galaxy takes the velocity of its
    # cell, which damps the effective f by ~sinc(k dz / 2) on coarse grids
    assert abs(cat - kaiser(f)) < 0.5 * abs(kaiser(f) - kaiser(a * f))


def test_redshift_space_density_is_deprecated(make_box):
    box = make_box(nsamp=16)
    d = box.realise_density()
    vz = fft.ifftn(box.realise_velocity(delta_x=d)[2]).real
    with pytest.warns(DeprecationWarning):
        box.redshift_space_density(delta_x=d, velocity_z=vz, sigma_nl=0.)


@pytest.mark.parametrize("bias", [1.0, 1.4])
def test_galaxy_mesh_carries_input_bias(make_box, bias):
    """GalaxyTracer reproduces the clustering of the (pre-biased) input field."""
    box = make_box(nsamp=32)
    P = Power(box)
    d = smoothed_field(box)
    nbar = 5e-3
    gal = GalaxyTracer(box, nbar)
    k, pfield, _ = P.estimate_power_spectrum(d)
    pg = np.mean([P.estimate_power_spectrum(gal.generate_mesh(bias * d, method='NGP',
                                                              overdensity=True))[1]
                         for _ in range(3)], 0)
    lo = k < 0.06
    measured = np.nanmean(((pg - 1. / nbar) / pfield)[lo])
    assert abs(measured - bias**2) < 0.1 * bias**2


@pytest.mark.parametrize("mas", ["NGP", "CIC"])
def test_galaxy_mesh_window(make_box, mas):
    """The cross power of a mesh with its input field shows the exact kernel."""
    box = make_box(nsamp=32)
    P = Power(box)
    d = smoothed_field(box)
    gal = GalaxyTracer(box, 5e-3)
    k, pf, _ = P.estimate_power_spectrum(d)
    pc = np.mean([P.estimate_power_spectrum(d, gal.generate_mesh(d, method=mas,
                                                                 overdensity=True))[1]
                         for _ in range(4)], 0)
    expected = 1. if mas == 'NGP' else shell_bin(box, cic_kernel(box))
    hi = k > 0.06
    assert abs(np.nanmean((pc / pf)[hi]) - np.nanmean(np.broadcast_to(expected, k.shape)[hi])) < 0.05


def test_generate_hi_mock():
    mock = fastbox.tracers.generate_hi_mock(fastbox.box.default_cosmo,
                                            box_scale=(1000., 800., 600.),
                                            nsamp=(20, 16, 12), redshift=0.8)
    assert mock.shape == (20, 16, 12)
    assert np.all(np.isfinite(mock)) and np.all(mock > 0.)


def test_halo_catalogue_noncubic(make_box):
    """Halo positions fill the whole box on a non-cubic grid."""
    box = make_box(box_scale=(800., 600., 1000.), nsamp=(20, 15, 25))
    hd = HaloDistribution(box, mass_range=(1e12, 1e14), mass_bins=3)
    counts = np.random.poisson(2., size=(box.Nx, box.Ny, box.Nz))
    cat = hd.realise_halo_catalogue(counts, scatter=True)
    assert len(cat) == counts.sum()
    L = np.array([box.Lx, box.Ly, box.Lz])
    assert np.all(cat >= 0.) and np.all(cat <= L)
    assert np.all(cat.max(0) > 0.95 * L)
