"""Export tracked JPEG comparisons of BLOS, transverse field and azimuth.

Field, inclination and azimuth FITS segments are needed. A fixed surface patch
is sampled in Carrington coordinates without an anchor epoch. SPINNME uses
each frame time and observer geometry interpolated between actual HMI epochs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import frames
from sunpy.map import Map

from _common import _image, _header, _model_time, _record_key, _sample_image, _series


@dataclass
class Record:
    field: Path
    inclination: Path
    azimuth: Path
    epoch: Time
    header: object
    shape: tuple


def index_reference(directory):
    """Read segment headers once; pair by archive record identity."""
    directory = Path(directory)
    inclinations = {}
    for path in sorted(directory.glob('*.inclination.fits')):
        header, shape = _header(path)
        key = _record_key(header)
        if key in inclinations:
            raise ValueError(f'Duplicate inclination record: {key}')
        inclinations[key] = (path, header, shape)
    records, seen = [], set()
    for path in sorted(directory.glob('*.field.fits')):
        header, shape = _header(path)
        _series(header, path, 'hmi.B_720s')
        key = _record_key(header)
        if key in seen:
            raise ValueError(f'Duplicate field record: {key}')
        seen.add(key)
        if key not in inclinations:
            raise ValueError(f'Missing inclination for {path}')
        inc, inc_header, inc_shape = inclinations[key]
        _series(inc_header, inc, 'hmi.B_720s')
        if abs((_model_time(inc_header)-_model_time(header)).to_value(u.s)) > 1:
            raise ValueError(f'Field/inclination epochs differ by more than 1 s: {path}')
        if inc_shape != shape:
            raise ValueError(f'Field/inclination shapes differ: {path}')
        for segment, own, expected in [('field', header, u.G), ('inclination', inc_header, u.deg)]:
            unit = str(own.get('BUNIT', '')).strip()
            if unit.lower() in ('', 'none', 'undefined', 'unknown', 'n/a', '?'):
                warnings.warn(f'Assuming standard HMI units for {segment} in {path}', RuntimeWarning)
            else:
                unit = {'gauss': 'G', 'degrees': 'deg', 'degree': 'deg'}.get(unit.lower(), unit)
                if not np.isclose(u.Unit(unit).to(expected), 1.):
                    raise ValueError(f'Expected native HMI {expected} units for {segment}: {path}')
        azimuth = path.with_name(path.name.removesuffix('.field.fits')+'.azimuth.fits')
        if not azimuth.exists():
            raise ValueError(f'Missing azimuth for {path}')
        azi_header, azi_shape = _header(azimuth)
        _series(azi_header, azimuth, 'hmi.B_720s')
        from astropy.wcs import WCS
        if azi_shape != shape or not WCS(header).celestial.wcs.compare(WCS(azi_header).celestial.wcs):
            raise ValueError(f'Field/azimuth grids differ: {path}')
        if _record_key(azi_header) != key or abs((_model_time(azi_header)-_model_time(header)).to_value(u.s)) > 1:
            raise ValueError(f'Field/azimuth records or epochs differ: {path}')
        unit = str(azi_header.get('BUNIT') or 'deg').strip()
        unit = {'degrees': 'deg', 'degree': 'deg'}.get(unit.lower(), unit)
        if not np.isclose(u.Unit(unit).to(u.deg), 1.):
            raise ValueError(f'Expected native HMI degrees for azimuth: {azimuth}')
        records.append(Record(path, inc, azimuth, _model_time(header).utc, header, shape))
    if not records:
        raise FileNotFoundError(f'No *.field.fits in {directory}')
    records.sort(key=lambda record: record.epoch.unix)
    if np.any(np.diff([record.epoch.unix for record in records]) <= 0):
        raise ValueError('Reference epochs must be unique.')
    return records


def frame_times(start, end, cadence):
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError('cadence must be finite and positive.')
    start, end = Time(start), Time(end)
    duration = (end-start).to_value(u.s)
    if duration < 0:
        raise ValueError('end must not precede start.')
    return start+np.arange(int(np.floor(duration/cadence+1e-8))+1)*cadence*u.s


def nearest_indices(records, times):
    """Earlier reference wins exact midpoint ties."""
    epochs = np.array([record.epoch.unix for record in records])
    target = np.asarray(times.unix)
    right = np.clip(np.searchsorted(epochs, target), 0, len(epochs)-1)
    left = np.maximum(right-1, 0)
    return np.where(target-epochs[left] <= epochs[right]-target, left, right)


def reference_map(record):
    header = record.header.copy()
    from spinnme.data.hmi_time import hmi_coordinate_header
    header = hmi_coordinate_header(header)
    return Map(np.broadcast_to(np.zeros((1, 1), dtype=np.uint8), record.shape), header)


def surface_patch(records, center, shape=(128, 256), pixel_scale=None):
    """Fixed tangent-plane surface grid, independent of any reference epoch."""
    if len(shape) != 2 or any(isinstance(v, bool) or int(v) != v or v < 2 for v in shape):
        raise ValueError('Patch height/width must be integers >= 2.')
    if len(center) != 2 or not np.isfinite(center).all() or not -90 <= center[1] <= 90:
        raise ValueError('Center requires finite Carrington longitude and latitude in [-90,90].')
    maps = [reference_map(record) for record in records]
    scale = np.median([abs(map_.scale.axis1).to_value(u.arcsec/u.pix) for map_ in maps]) if pixel_scale is None else pixel_scale
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('pixel_scale must be finite and positive.')
    # Convert apparent disk-center pixel size into a surface angular step.
    solar_radius = np.median([np.arcsin((map_.rsun_meters/map_.dsun).to_value(u.one))*u.rad.to(u.arcsec) for map_ in maps])
    step = scale/solar_radius
    lon, lat = np.deg2rad(center)
    normal = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    yy, xx = np.indices(shape, dtype=float)
    xx, yy = (xx-(shape[1]-1)/2)*step, (yy-(shape[0]-1)/2)*step
    vectors = normal+xx[..., None]*east+yy[..., None]*north
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    radius = np.median([map_.rsun_meters.to_value(u.m) for map_ in maps])
    return (np.rad2deg(np.arctan2(vectors[..., 1], vectors[..., 0])) % 360,
            np.rad2deg(np.arcsin(vectors[..., 2])), np.full(shape, radius))


def frame_record(records, epoch):
    """Interpolate observing geometry in time; never snap to the nearest record."""
    from spinnme.data.util import image_p_angle
    epochs = np.array([record.epoch.unix for record in records])
    time = Time(epoch)
    right = int(np.clip(np.searchsorted(epochs, time.unix), 0, len(records)-1))
    left = max(right-1, 0)
    if time.unix >= epochs[-1]:
        left = right = len(records)-1
    fraction = 0. if left == right else (time.unix-epochs[left])/(epochs[right]-epochs[left])
    fraction = float(np.clip(fraction, 0, 1))
    before, after = reference_map(records[left]), reference_map(records[right])
    header = before.fits_header.copy()
    def interpolate(a, b, *, angle=False):
        delta = (b-a+180) % 360-180 if angle else b-a
        return a+fraction*delta
    header['CRLN_OBS'] = interpolate(before.carrington_longitude.deg, after.carrington_longitude.deg, angle=True) % 360
    header['CRLT_OBS'] = interpolate(before.carrington_latitude.deg, after.carrington_latitude.deg)
    header['DSUN_OBS'] = interpolate(before.dsun.to_value(u.m), after.dsun.to_value(u.m))
    header['RSUN_REF'] = interpolate(before.rsun_meters.to_value(u.m), after.rsun_meters.to_value(u.m))
    # Avoid alternate observer/time metadata overriding the interpolated values.
    for key in ('HGLN_OBS', 'HGLT_OBS', 'HAEX_OBS', 'HAEY_OBS', 'HAEZ_OBS',
                'HEEX_OBS', 'HEEY_OBS', 'HEEZ_OBS', 'T_OBS', 'DATE-AVG'):
        header.remove(key, ignore_missing=True)
    angle = np.deg2rad(interpolate(np.rad2deg(-image_p_angle(before)),
                                   np.rad2deg(-image_p_angle(after)), angle=True))
    for key in list(header):
        if key.startswith(('PC1_', 'PC2_', 'CD1_', 'CD2_')):
            header.remove(key)
    header['PC1_1'], header['PC1_2'] = np.cos(angle), -np.sin(angle)
    header['PC2_1'], header['PC2_2'] = np.sin(angle), np.cos(angle)
    header['CUNIT1'], header['CUNIT2'] = 'deg', 'deg'
    header['CRVAL1'] = interpolate(before.reference_coordinate.Tx.to_value(u.deg),
                                    after.reference_coordinate.Tx.to_value(u.deg), angle=True)
    header['CRVAL2'] = interpolate(before.reference_coordinate.Ty.to_value(u.deg),
                                    after.reference_coordinate.Ty.to_value(u.deg))
    header['CDELT1'] = abs(interpolate(before.scale.axis1.to_value(u.deg/u.pix), after.scale.axis1.to_value(u.deg/u.pix)))
    header['CDELT2'] = abs(interpolate(before.scale.axis2.to_value(u.deg/u.pix), after.scale.axis2.to_value(u.deg/u.pix)))
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        if key in before.meta and key in after.meta:
            header[key] = interpolate(before.meta[key], after.meta[key])
    header['DATE-OBS'], header['MJD-OBS'], header['TIMESYS'] = time.utc.isot, time.utc.mjd, 'UTC'
    header['DATE-AVG'] = time.utc.isot
    header['T_OBS'] = time.tai.isot.replace('-', '.').replace('T', '_') + '_TAI'
    return Record(records[left].field, records[left].inclination, records[left].azimuth,
                  time.utc, header, records[left].shape)


def prepare_geometry(record, frozen, snapshot, instrument_id=None):
    from spinnme.grids import GridGeometry
    source = reference_map(record)
    coordinates = SkyCoord(frozen[0].ravel()*u.deg, frozen[1].ravel()*u.deg,
        frozen[2].ravel()*u.m, frame=frames.HeliographicCarrington,
        observer=source.observer_coordinate, obstime=source.reference_date)
    row = snapshot.grid({'id': 'video-frame', 'instrument_id': instrument_id or snapshot.instrument_id,
        'shape': list(record.shape), 'header': dict(source.fits_header),
        'time': {'model_time': record.epoch.utc.isot}}, instrument_id=instrument_id)
    geometry = GridGeometry(row).points(coordinates, Rs_per_ds=snapshot.Rs_per_ds, normalized_time=0.)
    return geometry, row, coordinates


def sample_azimuth(path, target_x, target_y):
    """Bilinear axial-angle interpolation, avoiding the 0/180 degree seam."""
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates
    header, shape = _header(path)
    wcs = WCS(header).celestial
    x, y = wcs.world_to_pixel_values(
        (target_x*u.arcsec).to_value(u.Unit(wcs.world_axis_units[0])),
        (target_y*u.arcsec).to_value(u.Unit(wcs.world_axis_units[1])))
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= shape[1]-1) & (y >= 0) & (y <= shape[0]-1)
    result = np.full(x.shape, np.nan)
    if not inside.any():
        return result
    x0, x1 = int(np.floor(x[inside].min())), int(np.ceil(x[inside].max()))+1
    y0, y1 = int(np.floor(y[inside].min())), int(np.ceil(y[inside].max()))+1
    angles = _image(path, (slice(y0, y1), slice(x0, x1)))
    valid = np.isfinite(angles) & (angles >= 0) & (angles <= 360)
    radians = np.deg2rad(2*np.where(valid, angles, 0))
    points = [y[inside]-y0, x[inside]-x0]
    def sample(values):
        return map_coordinates(values, points, order=1, mode='constant', cval=0, prefilter=False)
    cosine = sample(np.where(valid, np.cos(radians), 0))
    sine = sample(np.where(valid, np.sin(radians), 0))
    supported = (sample(valid.astype(float)) >= .999) & (np.hypot(cosine, sine) > 1e-8)
    angle = np.rad2deg(np.arctan2(sine, cosine))/2 % 180
    result[inside] = np.where(supported, angle, np.nan)
    return result


def prepare_reference(record, frozen, snapshot, instrument_id=None):
    """Cache one HMI map on the fixed surface grid at its actual epoch."""
    source = reference_map(record)
    geometry, row, coordinates = prepare_geometry(record, frozen, snapshot, instrument_id)
    hpc = coordinates.transform_to(source.coordinate_frame)
    x, y = hpc.Tx.to_value(u.arcsec), hpc.Ty.to_value(u.arcsec)
    reference = {key: _sample_image(record.field, x, y, limits=(0, np.inf),
                                    magnetic=(record.inclination, parameter, 1))
                 for key, parameter in (('B_los', 'apparent_B_los'), ('B_transverse', 'B_transverse'))}
    reference['azimuth'] = sample_azimuth(record.azimuth, x, y)
    visible = np.isfinite(geometry['coords']).all(-1) & (geometry['mu'].ravel() > 0)
    for key, values in reference.items():
        values[~visible] = np.nan
        reference[key] = values.reshape(frozen[0].shape)
    return reference, geometry, row, coordinates


def evaluate_magnetic(snapshot, geometry, epoch, instrument_id, batch_size=8192):
    """Only evaluate the field heads and project B; no Stokes synthesis.

    Uses the same Cartesian -> spherical -> image matrices as Snapshot.evaluate.
    Geometry stays on the CPU and only a batch is transferred to the device.
    Gradients remain enabled for vector-potential models, which differentiate A.
    """
    import torch
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    valid = np.isfinite(geometry['coords']).all(-1) & (geometry['mu'].ravel() > 0)
    output = {key: np.full(len(valid), np.nan) for key in ('B_los', 'B_transverse', 'azimuth')}
    indices = np.flatnonzero(valid)
    normalized_time = (Time(epoch)-Time(snapshot.ref_time)).to_value(u.s)/snapshot.seconds_per_dt
    vector_potential = bool(getattr(snapshot.parameter_model, 'vector_potential', False))
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        coords = geometry['coords'][selected].copy()
        coords[:, -1] = normalized_time
        coords = torch.as_tensor(coords, dtype=torch.float32, device=snapshot.device)
        c2s = torch.as_tensor(geometry['cartesian_to_spherical_transform'][selected], dtype=torch.float32, device=snapshot.device)
        projection = torch.as_tensor(geometry['rtp_to_img_transform'][selected], dtype=torch.float32, device=snapshot.device)
        with torch.set_grad_enabled(vector_potential):
            coords.requires_grad_(vector_potential)
            raw = snapshot.parameter_model(coords, instrument_id=instrument_id)
            field = torch.cat([raw[key] for key in ('b_x', 'b_y', 'b_z')], dim=-1)
            spherical = torch.einsum('nij,nj->ni', c2s, field)
            image = torch.einsum('nij,nj->ni', projection, spherical)*snapshot.gauss_per_dB
            transverse = torch.linalg.vector_norm(image[:, :2], dim=-1)
            azimuth = torch.remainder(torch.rad2deg(torch.atan2(-image[:, 0], image[:, 1])), 180)
        output['B_los'][selected] = image[:, 2].detach().cpu().numpy()
        output['B_transverse'][selected] = transverse.detach().cpu().numpy()
        output['azimuth'][selected] = np.where(transverse.detach().cpu().numpy() > 0,
                                                azimuth.detach().cpu().numpy(), np.nan)
    return output


def evaluate_blos(snapshot, geometry, epoch, instrument_id, batch_size=8192):
    return evaluate_magnetic(snapshot, geometry, epoch, instrument_id, batch_size)['B_los']


def azimuth_difference(prediction, reference):
    return (prediction-reference+90) % 180-90


def panel_styles(vmax, difference_max):
    import cmasher as cmr
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    return {
        'B_los': (SymLogNorm(linthresh=1., vmin=-vmax, vmax=vmax), 'RdBu_r',
                  SymLogNorm(linthresh=1., vmin=-difference_max, vmax=difference_max), 'G'),
        'B_transverse': (LogNorm(vmin=1., vmax=vmax, clip=True), 'viridis',
                         SymLogNorm(linthresh=1., vmin=-difference_max, vmax=difference_max), 'G'),
        'azimuth': (Normalize(0, 180), cmr.infinity, Normalize(-90, 90), 'deg'),
    }


def generate(input90, reference720, output, center_carrington, start, end, *, shape=(128, 256),
             cadence=10., pixel_scale=None, vmax=1500.,
             difference_max=300., max_reference_age=360., batch_size=8192,
             device='auto', instrument_id=None, dpi=120):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spinnme import load

    for key, value in [('vmax', vmax), ('difference_max', difference_max),
                       ('max_reference_age', max_reference_age), ('dpi', dpi)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if vmax <= 1:
        raise ValueError('vmax must exceed the transverse log minimum of 1 G.')
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Use a new or empty frame directory: {output}')
    records = index_reference(reference720)
    times = frame_times(start, end, cadence)
    assignments = nearest_indices(records, times)
    frozen = surface_patch(records, center_carrington, shape, pixel_scale)
    ages = np.array([(time-records[int(i)].epoch).to_value(u.s) for time, i in zip(times, assignments)])
    if np.any(abs(ages) > max_reference_age):
        warnings.warn('Some frames exceed --max-reference-age; the closest reference is retained and labeled REFERENCE GAP.', RuntimeWarning)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), layout='constrained')
    images = {}
    labels = {'B_los': 'BLOS', 'B_transverse': 'Transverse field', 'azimuth': 'Azimuth mod 180°'}
    for row, (key, (norm, cmap, difference_norm, unit)) in enumerate(panel_styles(vmax, difference_max).items()):
        images[key] = []
        for column, title in enumerate(('HMI 720 s', 'SPINNME 90 s', 'SPINNME − HMI')):
            ax = axes[row, column]
            image = ax.imshow(np.full(shape, np.nan), origin='lower', interpolation='nearest',
                              cmap=cmap if column < 2 else 'RdBu_r', norm=norm if column < 2 else difference_norm)
            images[key].append(image)
            ax.set_title(title+' · '+labels[key])
            if row == 2:
                ax.set_xlabel('Tracked patch x [pixel]')
            if column == 0:
                ax.set_ylabel('Tracked patch y [pixel]')
            bar = fig.colorbar(image, ax=ax, label=unit, shrink=.8)
            if key == 'azimuth':
                bar.set_ticks([0, 45, 90, 135, 180] if column < 2 else [-90, -45, 0, 45, 90])
    heading = fig.suptitle('')
    previous = None
    try:
        with load(input90, device=device) as snapshot:
            instrument_id = instrument_id or snapshot.instrument_id
            if instrument_id is None:
                raise ValueError('Select --instrument for a multi-instrument snapshot.')
            observations = [row for row in snapshot.observations if row['instrument_id'] == instrument_id]
            training = [row for row in observations if row.get('role') == 'train'] or observations
            training_times = [Time(row['time']['model_time']) for row in training]
            if training_times and (times[0] < min(training_times) or times[-1] > max(training_times)):
                warnings.warn('Requested epochs extend outside saved training times; model extrapolation is included.', RuntimeWarning)
            for number, (epoch, reference_index, age) in enumerate(zip(times, assignments, ages)):
                record = records[int(reference_index)]
                if previous != reference_index:
                    reference, _, _, _ = prepare_reference(record, frozen, snapshot, instrument_id)
                    previous = reference_index
                observing_record = frame_record(records, epoch)
                geometry, _, _ = prepare_geometry(observing_record, frozen, snapshot, instrument_id)
                from spinnme.data.util import image_p_angle
                # Compare azimuths in the continuously sampled frame's detector basis.
                roll_offset = np.rad2deg(image_p_angle(reference_map(observing_record))-
                                         image_p_angle(reference_map(record)))
                prediction = {key: values.reshape(shape) for key, values in
                              evaluate_magnetic(snapshot, geometry, epoch, instrument_id, batch_size).items()}
                fresh = bool(abs(age) <= max_reference_age)
                for key, panels in images.items():
                    shown, predicted = reference[key], prediction[key]
                    if key == 'azimuth':
                        shown = (shown+roll_offset) % 180
                    common = np.isfinite(predicted) & np.isfinite(shown)
                    residual = azimuth_difference(predicted, shown) if key == 'azimuth' else predicted-shown
                    difference = np.where(common, residual, np.nan)
                    for image, values in zip(panels, (shown, predicted, difference)):
                        image.set_data(np.ma.masked_invalid(values))
                heading.set_text(f'{epoch.utc.isot} UTC · Carrington ({center_carrington[0]:.3f}°, {center_carrington[1]:.3f}°)\n'
                    f'HMI {record.epoch.utc.isot} UTC · Δt={age:+.1f} s'+(' · REFERENCE GAP' if not fresh else ''))
                fig.savefig(output/f'frame_{number:06d}.jpg', dpi=dpi)
                if number % 50 == 0 or number == len(times)-1:
                    print(f'Frame {number+1}/{len(times)}', flush=True)
    finally:
        plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input90', required=True, type=Path)
    parser.add_argument('--reference720', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New or empty directory for JPEG frames')
    parser.add_argument('--center-carrington', required=True, nargs=2, type=float, metavar=('LON', 'LAT'))
    parser.add_argument('--start', required=True, help='UTC start')
    parser.add_argument('--end', required=True, help='UTC inclusive end if on cadence')
    parser.add_argument('--shape', nargs=2, type=int, default=(256, 128), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--cadence', type=float, default=10., help='Model sampling interval in seconds')
    parser.add_argument('--pixel-scale', type=float, help='Disk-center-equivalent arcsec/pixel for the fixed surface grid; default median HMI scale')
    parser.add_argument('--vmax', type=float, default=1500., help='Fixed magnetic bound [G]: symmetric for BLOS, upper limit for transverse (min 1 G)')
    parser.add_argument('--difference-max', type=float, default=300., help='Fixed symmetric magnetic residual bound [G]; azimuth residual is fixed to ±90 deg')
    parser.add_argument('--max-reference-age', type=float, default=360., help='Flag reference gaps above this age [s]; still show closest frame')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--instrument', dest='instrument_id')
    parser.add_argument('--dpi', type=int, default=120)
    args = vars(parser.parse_args(argv))
    args['shape'] = tuple(reversed(args['shape']))
    generate(**args)


if __name__ == '__main__':
    main()
