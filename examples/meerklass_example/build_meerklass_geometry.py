"""Build a MeerKLASS deep-field geometry configuration from the supplied data release.

The generated configuration uses the native map grid, selects the 971--1023 MHz
science band from the actual frequency vector, and calculates a local tangent-
plane bounding box suitable for FastBox's Cartesian approximation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pyccl as ccl


DATA_PATH = Path('/home/bruno/Downloads/13_fast/meerklass_example/MeerKLASS_HI_IM_2021/MK_2021_maps.npz')
OUT_PATH = Path('/home/bruno/Downloads/13_fast/meerklass_example/geometry_data/meerklass_deep_field_geometry.json')
LINE_FREQ_MHZ = 1420.405752
SCIENCE_BAND_MHZ = (971.0, 1023.0)
COSMOLOGY = {
    'Omega_c': 0.265,
    'Omega_b': 0.049,
    'h': 0.6736,
    'n_s': 0.9649,
    'sigma8': 0.811,
    'transfer_function': 'boltzmann_camb',
}


def circular_mean_deg(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted circular mean of angles in degrees."""
    angle = np.deg2rad(values)
    mean = np.sum(weights * np.exp(1j * angle)) / np.sum(weights)
    return float(np.rad2deg(np.angle(mean)) % 360.0)


def tangent_plane_coordinates(ra_deg: np.ndarray, dec_deg: np.ndarray,
                              ra0_deg: float, dec0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculate gnomonic tangent-plane coordinates, in radians."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    delta_ra = (ra - ra0 + np.pi) % (2.0 * np.pi) - np.pi
    denominator = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(delta_ra)
    xi = np.cos(dec) * np.sin(delta_ra) / denominator
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(delta_ra)) / denominator
    return xi, eta


def pixel_pitch_from_grid(xi: np.ndarray, eta: np.ndarray, direction: int) -> tuple[float, float]:
    """Return the median native pixel pitch and FFT-cell span in tangent-plane radians.

    The data are provided in a ZEA projection. A gnomonic local plane is not
    exactly uniformly spaced across the full released map, so use the median
    separation of adjacent native pixels rather than incorrectly requiring a
    globally affine coordinate relation.
    """
    if direction == 0:
        separation = np.hypot(np.diff(xi, axis=0), np.diff(eta, axis=0))
        ncell = xi.shape[0]
    elif direction == 1:
        separation = np.hypot(np.diff(xi, axis=1), np.diff(eta, axis=1))
        ncell = xi.shape[1]
    else:
        raise ValueError('direction must be 0 or 1')
    pitch = float(np.median(separation))
    return pitch, float(ncell * pitch)


@dataclass
class MeerKLASSGeometry:
    source_file: str
    science_band_request_mhz: list[float]
    science_band_native_mhz: list[float]
    native_shape_frequency_x_y: list[int]
    fastbox_shape_x_y_z: list[int]
    frequency_channel_width_mhz: float
    central_frequency_mhz: float
    central_redshift: float
    redshift_range: list[float]
    ra_dec_tangent_centre_deg: list[float]
    transverse_pixel_size_deg: list[float]
    tangent_plane_span_deg: list[float]
    box_scale_mpc: list[float]
    voxel_size_mpc: list[float]
    footprint_mask_fraction: float
    hitmap_nonzero_fraction_science_band: float
    coordinate_model: str
    los_convention: str
    cosmology: dict


def main() -> None:
    cosmo = ccl.Cosmology(**COSMOLOGY)
    with np.load(DATA_PATH, allow_pickle=False) as data:
        freqs = data['nu'].astype(float)
        ra = data['map_ra'].astype(float)
        dec = data['map_dec'].astype(float)
        footprint = data['CF_mask'].astype(float) > 0
        hits = data['hits'].astype(float)
        full_shape = list(data['Tsky'].shape)

    selected = (freqs >= SCIENCE_BAND_MHZ[0]) & (freqs <= SCIENCE_BAND_MHZ[1])
    if np.count_nonzero(selected) < 2:
        raise RuntimeError('The requested science band did not contain enough native channels.')
    science_freqs = freqs[selected]
    science_hits = hits[selected]

    # Centre the local plane on the released footprint rather than the empty map bounding box.
    weights = footprint.astype(float)
    ra0 = circular_mean_deg(ra[footprint], weights[footprint])
    dec0 = float(np.average(dec[footprint], weights=weights[footprint]))
    xi, eta = tangent_plane_coordinates(ra, dec, ra0, dec0)
    dxi, span_x = pixel_pitch_from_grid(xi, eta, direction=0)
    deta, span_y = pixel_pitch_from_grid(xi, eta, direction=1)

    # A FastBox side length is N times the cell pitch. This covers half a voxel
    # beyond the first and final native pixel centres, as required by FFT cells.
    z = LINE_FREQ_MHZ / science_freqs - 1.0
    a = 1.0 / (1.0 + z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    dchi = float(np.median(np.abs(np.diff(chi))))
    chi_c = float(ccl.comoving_angular_distance(
        cosmo, 1.0 / (1.0 + (LINE_FREQ_MHZ / np.mean(science_freqs) - 1.0))
    ))
    nx, ny = footprint.shape
    nz = science_freqs.size
    box_scale = (chi_c * span_x, chi_c * span_y, nz * dchi)
    central_frequency = float(np.mean(science_freqs))
    central_redshift = float(LINE_FREQ_MHZ / central_frequency - 1.0)

    result = MeerKLASSGeometry(
        source_file=str(DATA_PATH),
        science_band_request_mhz=list(SCIENCE_BAND_MHZ),
        science_band_native_mhz=[float(science_freqs[0]), float(science_freqs[-1])],
        native_shape_frequency_x_y=full_shape,
        fastbox_shape_x_y_z=[int(nx), int(ny), int(nz)],
        frequency_channel_width_mhz=float(np.median(np.diff(science_freqs))),
        central_frequency_mhz=central_frequency,
        central_redshift=central_redshift,
        redshift_range=[float(np.min(z)), float(np.max(z))],
        ra_dec_tangent_centre_deg=[ra0, dec0],
        transverse_pixel_size_deg=[float(np.rad2deg(dxi)), float(np.rad2deg(deta))],
        tangent_plane_span_deg=[float(np.rad2deg(span_x)), float(np.rad2deg(span_y))],
        box_scale_mpc=[float(value) for value in box_scale],
        voxel_size_mpc=[float(box_scale[0] / nx), float(box_scale[1] / ny), float(dchi)],
        footprint_mask_fraction=float(np.mean(footprint)),
        hitmap_nonzero_fraction_science_band=float(np.mean(science_hits > 0)),
        coordinate_model='Local gnomonic tangent-plane approximation to released ZEA coordinate arrays.',
        los_convention='FastBox z index follows the ascending released frequency array; RSD power uses mu^2 so its sign is convention independent.',
        cosmology=COSMOLOGY,
    )
    OUT_PATH.write_text(json.dumps(asdict(result), indent=2) + '\n')
    print(json.dumps(asdict(result), indent=2))
    print(f'Wrote geometry configuration: {OUT_PATH}')


if __name__ == '__main__':
    main()
