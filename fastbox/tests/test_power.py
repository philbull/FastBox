"""Tests for the power spectrum estimators and forward models in fastbox.power."""
import numpy as np
import pytest

from fastbox.beams import GaussianBeamModel
from fastbox.power import Power
from fastbox.tests.conftest import cic_kernel, kmag, shell_bin


@pytest.mark.parametrize("box_scale, nsamp", [
    (1000., 32),
    ((1200., 800., 1000.), (48, 32, 40)),
    ((1500., 750., 1000.), 32),
])
def test_power_spectrum_matches_theory(make_box, theory, box_scale, nsamp):
    """Mean P(k) over realisations matches the grid-binned theory."""
    box = make_box(box_scale=box_scale, nsamp=nsamp, redshift=0.5)
    P = Power(box)
    nreal = 16
    pks = []
    for _ in range(nreal):
        k, pk, _ = P.unweighted_power(box.realise_density())
        pks.append(pk)
    pk, err = np.mean(pks, 0), np.std(pks, 0) / np.sqrt(nreal)
    _, pt, _ = P.model_obs_power_gal(*theory(box), 1., MAS=None, rsd=False)

    sel = k > 3 * k[0]  # the lowest bins hold too few modes
    assert abs(np.nanmean(pk[sel] / pt[sel]) - 1.) < 0.03
    # chi2 per bin; it scatters between ~1 and 3 with the seed, so 4 is safe
    assert np.nanmean(((pk - pt) / err)[sel]**2) < 4.


def test_box_kmax_within_grid(make_box):
    """box.kmax should not exceed the grid corner, so the top bins are filled.

    (Low-k log bins can still be empty on a small grid, where they are
    narrower than the mode spacing.)
    """
    box = make_box(nsamp=32)
    corner = np.pi * np.sqrt((box.Nx / box.Lx)**2 + (box.Ny / box.Ly)**2
                             + (box.Nz / box.Lz)**2)
    assert box.kmax <= 1.01 * corner
    box.realise_density()
    with np.errstate(all='ignore'):
        _, pk, _ = box.binned_power_spectrum(nbins=20)
    assert np.all(np.isfinite(pk[-10:]))


@pytest.mark.parametrize("weights", ["mask", "taper"])
def test_weighted_power_normalisation(make_box, weights):
    """Masks and smooth tapers leave the small-scale amplitude unchanged."""
    box = make_box(nsamp=32)
    P = Power(box)
    d = box.realise_density()
    if weights == "mask":
        w = np.ones_like(d)
        w[:box.Nx // 2] = 0.
    else:
        x = np.arange(box.Nx)
        w = (1. + 0.5 * np.cos(2. * np.pi * x / box.Nx))[:, None, None] \
            * np.ones_like(d)
    k, pu, _ = P.unweighted_power(d)
    _, pw, _ = P.weighted_power(d, w1=w)
    _, pc, _ = P.weighted_power(d, d, w1=w, w2=w)
    sel = k > 0.05
    assert abs(np.nanmean(pw[sel] / pu[sel]) - 1.) < 0.05
    assert np.allclose(pc, pw)


def test_2pcf_matches_brute_force(make_box):
    """est_2pcf equals a direct lattice-offset sum binned the same way."""
    N, L, R = 24, 300., 4
    box = make_box(box_scale=L, nsamp=N, redshift=0.5)
    d = box.realise_density()
    r, xi = Power(box).est_2pcf(d)

    dc, dx = d - d.mean(), L / N
    bins = np.arange(dx, L / 2 + dx, dx)
    sums, cnt = np.zeros(bins.size + 1), np.zeros(bins.size + 1)
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for l in range(-R, R + 1):
                b = np.digitize(dx * np.sqrt(i * i + j * j + l * l), bins)
                sums[b] += np.mean(dc * np.roll(dc, (i, j, l), (0, 1, 2)))
                cnt[b] += 1
    brute = sums[1:bins.size] / np.maximum(cnt[1:bins.size], 1)
    nb = int(np.sum(bins[1:] < R * dx))  # bins fully inside the offset cube
    assert nb >= 2
    assert np.allclose(xi[:nb], brute[:nb], rtol=1e-8)


def test_bispectrum_quadratic_field(make_box):
    """A quadratic field g + a(g^2 - <g^2>) has tree-level B = 6a<P P>."""
    N, L, alpha, nreal, nbins = 48, 800., 0.5, 10, 6
    box = make_box(box_scale=L, nsamp=N, redshift=0.5)
    P = Power(box)
    kf, kn = P._isotropic_k_range()
    edges = np.linspace(kf, kn, nbins + 1)
    ratios = []
    for _ in range(nreal):
        g = box.realise_density()
        g = 0.15 * g / g.std()
        q = g + alpha * (g**2 - np.mean(g**2))
        _, b_g, _ = P.bispectrum_equilateral(g, n_bins=nbins)
        _, b_q, _ = P.bispectrum_equilateral(q, n_bins=nbins)
        # <P(k1) P(k2)> over closed triangles in each shell, from the same field
        Pk = np.abs(np.fft.fftn(g))**2 * L**3 / N**6
        pred = []
        for i in range(nbins):
            m = (box.k >= edges[i]) & (box.k < edges[i + 1])
            I, Pm = np.fft.ifftn(m).real, np.fft.ifftn(m * Pk).real
            pred.append(6. * alpha * np.sum(Pm * Pm * I) / np.sum(I**3))
        ratios.append((b_q - b_g) / np.array(pred))
    ratios = np.array(ratios)
    mean, err = ratios.mean(0), ratios.std(0) / np.sqrt(nreal)
    assert np.all(np.abs(mean - 1.) < 3. * err + 0.03)


def test_statistics_run_on_noncubic_grid(make_box):
    """est_2pcf and bispectrum_equilateral handle unequal cell counts."""
    box = make_box(box_scale=(600., 400., 500.), nsamp=(24, 16, 20))
    P = Power(box)
    d = box.realise_density()
    r, xi = P.est_2pcf(d)
    assert np.all(np.isfinite(xi))
    _, b, _ = P.bispectrum_equilateral(d, n_bins=4)
    assert b.shape == (4,)


def test_forward_models_agree(make_box, theory):
    """With identical physics the IM, galaxy and cross models are identical."""
    box = make_box(nsamp=32)
    P = Power(box)
    km, pkm = theory(box, linear=True)
    for fog in ['lorentzian', 'gaussian']:
        im = P.model_obs_power_IM(km, pkm, 1., 1., 0., sigma_nl=300.,
                                  fog_model=fog)[1]
        gal = P.model_obs_power_gal(km, pkm, 1., MAS=None, sigma_nl=300.,
                                    fog_model=fog)[1]
        cc = P.model_obs_power_CC(km, pkm, 1., 1., 1., 0., MAS=None,
                                  sigma_nl=300., fog_model=fog)[1]
        assert np.allclose(gal, im, rtol=1e-12)
        assert np.allclose(cc, im, rtol=1e-12)


@pytest.mark.parametrize("model", ["IM", "gal", "CC"])
def test_fog_model_names(make_box, theory, model):
    """fog_model is case-insensitive, 'none' disables FoG, typos raise."""
    box = make_box(nsamp=16)
    P = Power(box)
    km, pkm = theory(box, linear=True)
    fn = {
        "IM": lambda m, s=300.: P.model_obs_power_IM(km, pkm, 1., 1., 0., sigma_nl=s, fog_model=m)[1],
        "gal": lambda m, s=300.: P.model_obs_power_gal(km, pkm, 1., MAS=None, sigma_nl=s, fog_model=m)[1],
        "CC": lambda m, s=300.: P.model_obs_power_CC(km, pkm, 1., 1., 1., 0., MAS=None, sigma_nl=s, fog_model=m)[1],
    }[model]
    assert np.allclose(fn('Lorentzian'), fn('lorentzian'))
    assert np.allclose(fn('none'), fn('lorentzian', s=0.))
    assert np.allclose(fn(None), fn('none'))
    with pytest.raises(ValueError):
        fn('cauchy')


@pytest.mark.parametrize("mas", ["NGP", "CIC"])
def test_galaxy_model_window_is_discrete_kernel(make_box, theory, mas):
    """model_obs_power_gal applies the exact window of a scattered catalogue."""
    box = make_box(nsamp=32)
    P = Power(box)
    km, pkm = theory(box)
    base = P.model_obs_power_gal(km, pkm, 1., MAS=None, rsd=False)[1]
    with_mas = P.model_obs_power_gal(km, pkm, 1., MAS=mas, rsd=False)[1]
    # The model applies the window mode by mode, so compare it binned the same
    # way, weighted by the power in each mode
    k3 = kmag(box)
    pk3 = np.interp(k3.flatten(), km, pkm).reshape(k3.shape)
    W2 = 1. if mas == 'NGP' else cic_kernel(box)**2
    expected = shell_bin(box, pk3 * W2) / shell_bin(box, pk3)
    assert np.allclose(with_mas / base, expected, rtol=1e-6, equal_nan=True)


def test_matter_power_spectrum_rsd_runs(make_box):
    box = make_box(nsamp=16)
    k = np.logspace(-3, 0, 50)
    _, pk = Power(box).matter_power_spectrum(k, rsd=True, sigma_nl=300.)
    assert np.all(np.isfinite(pk)) and np.all(pk > 0)


def test_gaussian_beam_matches_model(make_box):
    """GaussianBeamModel damps each mode exactly as model_obs_power_IM assumes."""
    import pyccl as ccl
    # 12.5 Mpc pixels, so the 0.3 deg beam (R ~ 15 Mpc) is resolved
    box = make_box(box_scale=(500., 300., 200.), nsamp=(40, 24, 16))
    sig = 0.3
    bc = GaussianBeamModel(box, sig).beam_cube()
    assert bc.shape == (box.Nx, box.Ny, box.Nz)

    # Beam is centred on the box
    ax_, ay_ = box.pixel_array()
    ix, iy = np.unravel_index(np.argmax(bc[..., 0]), bc.shape[:2])
    assert abs(ax_[ix]) < np.diff(ax_)[0] and abs(ay_[iy]) < np.diff(ay_)[0]

    # |B(k_perp)|^2 against the model's Gaussian damping, per mode
    b0 = bc[..., 0]
    b0 = np.roll(np.roll(b0, -(box.Nx // 2), 0), -(box.Ny // 2), 1)
    Bk2 = np.abs(np.fft.fft2(b0 / b0.sum()))**2
    R = np.radians(sig) * ccl.comoving_radial_distance(box.cosmo, box.scale_factor)
    kx = 2. * np.pi * np.fft.fftfreq(box.Nx, box.Lx / box.Nx)
    ky = 2. * np.pi * np.fft.fftfreq(box.Ny, box.Ly / box.Ny)
    model = np.exp(-(kx[:, None]**2 + ky[None, :]**2) * R**2)
    sel = model > 0.05
    assert np.allclose(Bk2[sel], model[sel], rtol=1e-3)
