"""End-to-end check of the HI x galaxy example in paper.md.

The paper's example code must produce maps that its own forward models
describe. A smaller grid is used here; the pipeline is otherwise the same,
except that the foregrounds are left out, so there is no PCA signal loss.
"""
import numpy as np

import fastbox
from fastbox.power import Power
from fastbox.tracers import GalaxyTracer


def run_example(make_box, nreal=3, N=40, L=2000.):
    auto, cross = [], []
    for _ in range(nreal):
        box = make_box(box_scale=(L, L, L), nsamp=N, redshift=0.8)
        box.realise_density()

        # As in paper.md: "Generating a simulation box"
        tracer = fastbox.tracers.HITracer(box)
        delta_hi = box.linear_rsd_density(box.delta_x, bias=tracer.bias_HI(),
                                          sigma_v=120.)
        delta_ln = box.lognormal(delta_hi)
        signal_cube = tracer.signal_amplitude() * (1. + delta_ln)

        # As in paper.md: "Galaxy catalogue and HI-galaxy cross spectrum"
        b_gal = 1.4
        delta_g = box.lognormal(box.linear_rsd_density(box.delta_x, bias=b_gal,
                                                       sigma_v=120.))
        gal = GalaxyTracer(box, vol_density=1e-3)
        delta_gal = gal.generate_mesh(delta_g, method='CIC', overdensity=True)

        power = Power(box)
        k, pa, _ = power.estimate_power_spectrum(signal_cube)
        _, pc, _ = power.estimate_power_spectrum(signal_cube, delta_gal)
        auto.append(pa)
        cross.append(pc)

    th_k = np.logspace(-4, 1, 2000)
    _, th_pk = power.predict_matter_power_spectrum(th_k)
    _, pk_im, _ = power.predict_hi_power_spectrum(th_k, th_pk, bias=tracer.bias_HI(),
                                                  Tb=tracer.signal_amplitude(),
                                                  sigdeg=0., sigma_nl=120.)
    _, pk_cc, _ = power.predict_cross_spectrum(th_k, th_pk,
                                               bias_HI=tracer.bias_HI(),
                                               bias_gal=b_gal,
                                               Tb=tracer.signal_amplitude(),
                                               sigdeg=0., MAS='CIC', sigma_nl=120.)
    return k, np.mean(auto, 0) / pk_im, np.mean(cross, 0) / pk_cc


def test_paper_example_matches_models(make_box):
    """On large scales the measured spectra match the forward models.

    Before the example was fixed, the cross spectrum came out at ~0.4 of
    the model. The log-normal transform adds some small-scale power that the
    linear models leave out, so only k < 0.05 / Mpc is compared.
    """
    k, r_auto, r_cross = run_example(make_box)
    lo = (k > 2 * k[0]) & (k < 0.05)
    assert abs(np.nanmean(r_auto[lo]) - 1.) < 0.15
    assert abs(np.nanmean(r_cross[lo]) - 1.) < 0.15
