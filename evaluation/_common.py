"""Shared FITS readers, masked metrics and figure output for the workflow plot scripts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from spinnme.evaluation import comparison as comparison, residual as residual
from spinnme.evaluation.metrics import broadcast_mask
from spinnme.plotting import save_figure as save_figure

_broadcast_mask = broadcast_mask

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def read_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only numeric arrays and JSON metadata; never allow object pickle payloads."""
    arrays, metadata = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if key in {"metadata", "metadata_json"}:
                raw = archive[key].item()
                if isinstance(raw, bytes):
                    raw = raw.decode()
                metadata.update(json.loads(str(raw)))
            elif archive[key].dtype.kind in "biufc":
                arrays[key] = np.asarray(archive[key])
    if isinstance(metadata.get("header"), str):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("Public FITS-header NPZ products require astropy; install the dependencies in requirements.txt.") from exc
        header = fits.Header.fromstring(metadata["header"], sep="\n")
        metadata["header"] = dict(header)
    return arrays, metadata


def _header(path):
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
        image = next((hdu for hdu in hdus if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                      and hdu.header.get('NAXIS') == 2), None)
        if image is None:
            raise ValueError(f"Expected a two-dimensional FITS segment: {path}")
        header = hdus[0].header.copy()
        header.update(image.header)
        shape = (image.header['NAXIS2'], image.header['NAXIS1'])
    return header, shape


def _image(path, slices):
    with fits.open(path, memmap=False) as hdus:
        image = next(hdu for hdu in hdus if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                     and hdu.header.get('NAXIS') == 2)
        return np.asarray(image.section[slices], dtype=float)


def _model_time(header):
    from spinnme.data.hmi_time import hmi_observation_time
    return Time(hmi_observation_time(header)).utc


def _record_key(header):
    # Pipeline and Stokes T_OBS may differ slightly. T_REC is the shared
    # archive record index, never used as the SPINNME model epoch.
    return str(header.get('T_REC', _model_time(header).utc.isot))


def _series(header, path, expected):
    declared = str(header.get('SERIES', header.get('JSOC_SER', ''))).lower()
    if expected.lower() not in (declared + ' ' + Path(path).name.lower()):
        raise ValueError(f"Cannot verify {expected} in SERIES/JSOC_SER or filename: {path}")


def _world(header, shape):
    wcs = WCS(header).celestial
    if list(wcs.wcs.ctype) != ['HPLN-TAN', 'HPLT-TAN']:
        raise ValueError("Comparison grid must use a verified helioprojective TAN WCS.")
    y, x = np.indices(shape)
    world = wcs.pixel_to_world_values(x, y)
    result = []
    for values, unit in zip(world, wcs.world_axis_units):
        degree = (np.asarray(values) * u.Unit(unit)).to_value(u.deg)
        result.append(((degree+180) % 360-180)*3600)
    return tuple(result)


def _sample_image(path, target_x, target_y, *, nearest=False, limits=None, confidence=None, accepted=None,
                  magnetic=None):
    header, shape = _header(path)
    wcs = WCS(header).celestial
    units = wcs.world_axis_units
    x, y = wcs.world_to_pixel_values((target_x*u.arcsec).to_value(u.Unit(units[0])),
                                    (target_y*u.arcsec).to_value(u.Unit(units[1])))
    valid = (x >= -1e-6) & (x <= shape[1]-1+1e-6) & (y >= -1e-6) & (y <= shape[0]-1+1e-6)
    x, y = np.clip(x, 0, shape[1]-1), np.clip(y, 0, shape[0]-1)
    if not valid.any():
        return np.full(x.shape, np.nan)
    x0, x1 = max(0, int(np.floor(x[valid].min()))), min(shape[1], int(np.ceil(x[valid].max()))+1)
    y0, y1 = max(0, int(np.floor(y[valid].min()))), min(shape[0], int(np.ceil(y[valid].max()))+1)
    slices = (slice(y0, y1), slice(x0, x1))
    data = _image(path, slices)
    native_valid = np.isfinite(data)
    if limits is not None:
        native_valid &= (data >= limits[0]) & (data <= limits[1])
    if confidence is not None:
        quality_header, quality_shape = _header(confidence)
        if quality_shape != shape or not WCS(header).celestial.wcs.compare(WCS(quality_header).celestial.wcs):
            raise ValueError("HMI confidence and parameter images must share the exact native WCS.")
        native_valid &= np.isin(_image(confidence, slices), accepted)
    if magnetic is not None:
        inclination_path, parameter, los_sign = magnetic
        inc_header, inc_shape = _header(inclination_path)
        if inc_shape != shape or not WCS(header).celestial.wcs.compare(WCS(inc_header).celestial.wcs):
            raise ValueError("HMI field and inclination must share the exact native WCS before nonlinear conversion.")
        inclination = _image(inclination_path, slices)
        native_valid &= np.isfinite(inclination) & (inclination >= 0) & (inclination <= 180)
        # Convert on the native grid. Interpolating B and gamma separately then
        # forming B*cos(gamma) invents flux in unresolved mixed-polarity pixels.
        data = _magnetic(data, inclination, los_sign)[parameter]
    result = map_coordinates(np.where(native_valid, data, 0), [y-y0, x-x0], order=0 if nearest else 1,
                             mode='constant', cval=0., prefilter=False)
    support = map_coordinates(native_valid.astype(float), [y-y0, x-x0], order=0 if nearest else 1,
                              mode='constant', cval=0., prefilter=False)
    result[~valid | (support < .999)] = np.nan
    return result


def _magnetic(strength, inclination, los_sign, filling_factor=1):
    if los_sign not in (-1, 1):
        raise ValueError("Explicit verified los_sign must be +1 or -1 for each archive.")
    angle = np.deg2rad(inclination)
    return {'apparent_B_los': los_sign*filling_factor*strength*np.cos(angle), 'B': strength,
            'B_transverse': abs(strength*np.sin(angle)),
            'inclination': inclination if los_sign == 1 else 180-inclination}
