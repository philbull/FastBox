"""JAX kernels for the Gaussian density, linear RSD, beam and power pipeline.

Prepare box-dependent inputs outside JAX transformations. Inside a transformed
function, pass those inputs as arrays and scalars, or supply them from a
differentiable cosmology model.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pyccl as ccl
from jax.ops import segment_sum

from .power import Power


class RSDInputs(NamedTuple):
    k_parallel: np.ndarray
    k_squared: np.ndarray
    growth_rate: float
    aH: float


class PowerBins(NamedTuple):
    centers: np.ndarray
    shell_ids: np.ndarray
    counts: np.ndarray


def prepare_rsd(box):
    """Prepare Fourier geometry and pyCCL background for the box redshift."""
    a = box.scale_factor
    H = 100.0 * box.cosmo["h"] * ccl.h_over_h0(box.cosmo, a)
    return RSDInputs(
        2.0 * np.pi * box.Kz / box.Lz,
        box.k**2,
        ccl.growth_rate(box.cosmo, a),
        a * H,
    )


def prepare_power_bins(box):
    """Prepare the same fixed Fourier shells as FastBox's power estimator."""
    edges = Power(box)._power_bin_edges()
    shell_ids = np.digitize(box.k.ravel(), edges)
    counts = np.bincount(shell_ids, minlength=edges.size + 1)
    return PowerBins(0.5 * (edges[1:] + edges[:-1]), shell_ids, counts)


def gaussian_density(white_noise, pk, boxfactor):
    """Filter unit Gaussian voxel noise by P(k), in Mpc^3, on the FFT grid.

    Supply a real draw from ``jax.random.normal`` or a fixed noise field.
    Use P(0) = 0 for a zero-mean density contrast. Zero-power modes have
    zero gradients by convention.
    """
    white_noise = jnp.asarray(white_noise)
    pk = jnp.asarray(pk)
    nonzero_power = pk != 0.0
    safe_pk = jnp.where(nonzero_power, pk, 1.0)
    amplitude = jnp.where(
        nonzero_power, jnp.sqrt(safe_pk * (boxfactor / white_noise.size)), 0.0
    )
    return jnp.fft.ifftn(jnp.fft.fftn(white_noise) * amplitude).real


def linear_rsd_density(field, rsd, bias=1.0, sigma_v=0.0, fog_model="lorentzian"):
    """Apply linear Kaiser RSD and FoG damping. Keep the model name static."""
    nonzero = rsd.k_squared > 0.0
    safe_k_squared = jnp.where(nonzero, rsd.k_squared, 1.0)
    mu_squared = jnp.where(nonzero, rsd.k_parallel**2 / safe_k_squared, 0.0)
    x = rsd.k_parallel * (jnp.maximum(sigma_v, 0.0) / rsd.aH)
    fog_model = fog_model.lower()
    if fog_model == "lorentzian":
        damping = 1.0 / jnp.sqrt(1.0 + x**2)
    elif fog_model == "gaussian":
        damping = jnp.exp(-0.5 * x**2)
    elif fog_model == "none":
        damping = 1.0
    else:
        raise ValueError("fog_model must be 'lorentzian', 'gaussian' or 'none'")
    transfer = (bias + rsd.growth_rate * mu_squared) * damping
    return jnp.fft.ifftn(jnp.fft.fftn(field) * transfer).real


def convolve_gaussian_beam(field, ang_x, ang_y, sigma_deg):
    """Apply the periodic Gaussian beam on the two transverse axes."""
    ang_x, ang_y = jnp.asarray(ang_x), jnp.asarray(ang_y)
    radius_squared = ang_x[:, None]**2 + ang_y[None, :]**2
    beam = jnp.exp(-0.5 * radius_squared / sigma_deg**2)
    beam /= jnp.sum(beam)
    axes = (0, 1)
    beam_k = jnp.fft.fft2(jnp.fft.ifftshift(beam, axes=axes), axes=axes)
    field_k = jnp.fft.fft2(field, axes=axes)
    return jnp.fft.ifft2(field_k * beam_k[:, :, None], axes=axes).real


def estimate_power_spectrum(field, bins, boxfactor, field2=None, w1=None, w2=None):
    """Return bin centers, weighted auto-/cross-power and standard errors.

    Weights use the same mean-product normalisation as the NumPy estimator.
    Empty bins return NaN. Shell membership is fixed during differentiation.
    """
    weight1 = 1.0 if w1 is None else jnp.asarray(w1)
    modes = jnp.fft.fftn(field * weight1)
    if field2 is None:
        other_modes = modes
        weight2 = weight1
    else:
        weight2 = 1.0 if w2 is None else jnp.asarray(w2)
        other_modes = jnp.fft.fftn(field2 * weight2)
    mode_power = (modes * jnp.conj(other_modes)).real.ravel()
    mode_power /= boxfactor * jnp.mean(weight1 * weight2)

    counts = jnp.asarray(bins.counts)
    safe_counts = jnp.maximum(counts, 1)
    means = segment_sum(mode_power, bins.shell_ids, num_segments=counts.size) / safe_counts
    residuals = mode_power - means[bins.shell_ids]
    squared_residuals = segment_sum(
        residuals**2, bins.shell_ids, num_segments=counts.size
    )
    nonzero_variance = squared_residuals != 0.0
    # Avoid differentiating sqrt at zero when all modes in a shell agree.
    safe_variance = jnp.where(nonzero_variance, squared_residuals, 1.0)
    errors = jnp.where(nonzero_variance, jnp.sqrt(safe_variance) / safe_counts, 0.0)
    populated = counts[1:-1] > 0
    return (
        jnp.asarray(bins.centers),
        jnp.where(populated, means[1:-1], jnp.nan),
        jnp.where(populated, errors[1:-1], jnp.nan),
    )
