"""Compare Hinode MERLIN and HMI with SPINNME HMI and Hinode inversions.

The Hinode inversion supplies the native sky grid, slit times and observer.
HMI references and both SPINNME predictions use the same aligned Hinode grid.
MERLIN retains its native pixels in that common sky frame.
Hinode is evaluated at its saved unshifted coordinates, then displayed in the
aligned frame. HMI uses physical coordinates derived from that shifted frame.
Both retain the same native slit times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time

from _common import (save_figure,
                     _header, _model_time, _record_key, _series, _sample_image, _magnetic)

PARAMETERS = ('apparent_B_los', 'B', 'B_transverse', 'azimuth')
PLOT_PARAMETERS = ('apparent_B_los', 'B_transverse', 'azimuth')
SNAPSHOT = '/glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_v01/inversion.pme'
HMI_REFERENCE = '/glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s'
HINODE_RESULT = '/glade/work/rjarolim/spinn_me/hinode_sp_20240323_020035_v03/inversion.pme'
SOURCES = ('hmi', 'spinnme', 'hinode', 'spinnme_hinode')
HINODE_FILE = '/glade/work/rjarolim/data/hinode_merlin/20240323_020035/20240323_020035.fits'
START, END, TIME_SCALE = '2024-03-23T01:00:00', '2024-03-23T03:00:00', 'tai'
INSTRUMENT_ID = 'HMI'
MAXIMUM_TIME_OFFSET_SECONDS = 360
DEVICE, BATCH_SIZE = None, 8192
HINODE = {'file': HINODE_FILE, 'los_sign': 1, 'accepted_fit_status': None, 'azimuth_rotation_deg': 90.}
PLOT_X_MAX_ARCSEC = -110.


def _path(value):
    return Path(value).expanduser().resolve()


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return value


def hmi_records(directory):
    """Index existing native B_720s field maps and their inclination partners."""
    records = []
    for path in sorted(_path(directory).glob('*.field.fits')):
        inclination = path.with_name(path.name.removesuffix('.field.fits')+'.inclination.fits')
        azimuth = path.with_name(path.name.removesuffix('.field.fits')+'.azimuth.fits')
        if not inclination.exists() or not azimuth.exists():
            continue
        header, shape = _header(path)
        inc_header, _ = _header(inclination)
        azi_header, _ = _header(azimuth)
        for own, file in ((header, path), (inc_header, inclination), (azi_header, azimuth)):
            _series(own, file, 'hmi.B_720s')
        if any(int(own.get('QUALITY', 0)) != 0 for own in (header, inc_header, azi_header)):
            continue
        if any(_record_key(header) != _record_key(own) for own in (inc_header, azi_header)):
            raise ValueError(f'HMI field/inclination record mismatch: {path}')
        records.append({'time': _model_time(header).utc.isot, 'record_key': _record_key(header),
                        'field': str(path), 'inclination': str(inclination), 'azimuth': str(azimuth), 'header': header, 'shape': shape})
    records.sort(key=lambda row: Time(row['time']).unix)
    if not records:
        raise FileNotFoundError(f'No paired HMI B_720s field/inclination/azimuth files in {directory}.')
    if len({row['time'] for row in records}) != len(records):
        raise ValueError('HMI reference epochs must be unique.')
    return records


def time_window(start, end, time_scale):
    begin, finish = Time(start, scale=time_scale).unix, Time(end, scale=time_scale).unix
    if not np.isfinite([begin, finish]).all() or finish <= begin:
        raise ValueError('end must be later than start.')
    return begin, finish


def merlin(settings):
    """Read named MERLIN extensions, exact X/Y arrays, UTC slit times and f.

    The archive's extension 12 is the *magnetic* filling factor.
    Use finite physical values by default; an optional explicit fit-status
    selection can restrict the native maps.
    """
    path = _path(settings['file']).resolve()
    with fits.open(path, memmap=False) as hdus:
        if str(hdus[0].header.get('INVCODE', '')).strip().upper() != 'MERLIN':
            raise ValueError("The reference must be a Hinode MERLIN Level-2 inversion.")
        names = {'B': 'Field_Strength', 'inclination': 'Field_Inclination',
                 'azimuth': 'Field_Azimuth', 'f': 'Stray_Light_Fill_Factor',
                 'status': 'Fitting_Attribute', 'x': 'X_Coordinate', 'y': 'Y_Coordinate'}
        arrays = {key: np.asarray(hdus[name].data, dtype=float) for key, name in names.items()}
        raw_times = hdus['Times'].data
        if getattr(raw_times.dtype, 'names', None):
            if 'TIMES' not in raw_times.dtype.names:
                raise ValueError("MERLIN Times table needs its TIMES column.")
            raw_times = raw_times['TIMES']
        times = Time(np.asarray(raw_times).astype(str).reshape(-1), scale='utc').unix
    shape = arrays['B'].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays.values()):
        raise ValueError("All MERLIN parameter, status and X/Y arrays must have one shared (y,x) shape.")
    if len(times) != shape[1] or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("MERLIN Times must provide a strictly increasing UTC epoch per slit column.")
    for key in ('B', 'inclination', 'f'):
        if not np.isfinite(arrays[key]).any():
            raise ValueError(f"No finite MERLIN {key} values.")
    mask = np.logical_and.reduce([np.isfinite(array) for array in arrays.values()])
    mask &= (arrays['B'] >= 0) & (arrays['inclination'] >= 0) & (arrays['inclination'] <= 180)
    mask &= (arrays['f'] >= 0) & (arrays['f'] <= 1)
    mask &= np.isin(arrays['status'], range(1, 7))
    good = settings.get('accepted_fit_status')
    if good is not None:
        mask &= np.isin(arrays['status'], good)
    arrays.update(mask=mask, times=np.broadcast_to(times[None, :], shape).copy())
    return arrays


def pair_times(times, records, maximum_offset=MAXIMUM_TIME_OFFSET_SECONDS):
    """Pair each raster pixel to the nearest actual HMI epoch, without extrapolation."""
    maximum_offset = _positive(maximum_offset, 'maximum_time_offset_seconds')
    epochs = np.asarray([Time(row['time']).unix for row in records])
    flat = np.asarray(times).ravel()
    insertion = np.clip(np.searchsorted(epochs, flat), 0, len(epochs)-1)
    before = np.maximum(insertion-1, 0)
    index = np.where(abs(epochs[before]-flat) <= abs(epochs[insertion]-flat), before, insertion)
    offset = epochs[index]-flat
    valid = np.isfinite(flat) & (abs(offset) <= maximum_offset)
    valid &= (flat >= epochs[0]) & (flat <= epochs[-1])
    return index.reshape(times.shape), offset.reshape(times.shape), valid.reshape(times.shape)


def native_crop(source, begin, finish, bounds=None):
    """Keep native Hinode pixels; crop only the enclosing array rectangle."""
    keep = np.isfinite(source['x']) & np.isfinite(source['y'])
    keep &= (source['times'] >= begin) & (source['times'] <= finish)
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        if not np.isfinite(bounds).all() or xmin >= xmax or ymin >= ymax:
            raise ValueError('hpc-range must contain xmin,xmax,ymin,ymax.')
        keep &= (source['x'] >= xmin) & (source['x'] <= xmax) & (source['y'] >= ymin) & (source['y'] <= ymax)
    rows, cols = np.where(keep)
    if not len(rows):
        raise ValueError('No Hinode coordinates lie inside the requested time/sky interval.')
    slices = (slice(rows.min(), rows.max()+1), slice(cols.min(), cols.max()+1))
    shape = source['mask'].shape
    result = {key: value[slices].copy() if isinstance(value, np.ndarray) and value.shape[:2] == shape
              else value for key, value in source.items()}
    result['mask'] &= keep[slices]
    return result


def sample_snapshot(model, inversion, batch_size, *, image_roll=None, instrument_id=INSTRUMENT_ID):
    """Evaluate the HMI field at saved Hinode xyzt, changing only normalization.

    The Hinode projection matrices define the common LOS. Image roll can change
    the displayed azimuth basis; it cannot change the model's sampling positions.
    """
    import torch
    from types import SimpleNamespace
    from spinnme.progress import progress

    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    shape = inversion['mask'].shape
    coords = inversion['coords'].reshape(-1, 4).copy()
    coords[:, :3] *= inversion['Rs_per_ds']/model.Rs_per_ds
    delta_seconds = (Time(inversion['ref_time'])-Time(model.ref_time)).to_value(u.s)
    coords[:, 3] = (coords[:, 3]*inversion['seconds_per_dt']+delta_seconds)/model.seconds_per_dt
    c2s = inversion['cartesian_to_spherical_transform'].reshape(-1, 3, 3)
    projection = inversion['rtp_to_img_transform'].reshape(-1, 3, 3)
    native_roll = inversion['image_p_angle'].reshape(-1)
    target_roll = np.broadcast_to(0. if image_roll is None else image_roll, shape).reshape(-1)
    valid = inversion['mask'].ravel() & np.isfinite(coords).all(-1)
    valid &= np.isfinite(c2s).all((1, 2)) & np.isfinite(projection).all((1, 2))
    indices = np.flatnonzero(valid)
    output = {key: np.full(len(coords), np.nan) for key in ('B', 'B_los', 'B_transverse', 'azimuth')}
    vector_potential = bool(getattr(model.parameter_model, 'vector_potential', False))
    for start in progress(range(0, len(indices), batch_size), desc='Sampling SPINNME'):
        selected = indices[start:start+batch_size]
        def tensor(value):
            return torch.as_tensor(value[selected], dtype=torch.float32, device=model.device)
        points = tensor(coords)
        with torch.set_grad_enabled(vector_potential):
            points.requires_grad_(vector_potential)
            raw = model.parameter_model(points, instrument_id=instrument_id)
            field = torch.cat([raw[key] for key in ('b_x', 'b_y', 'b_z')], dim=-1)*model.gauss_per_dB
            spherical = torch.einsum('nij,nj->ni', tensor(c2s), field)
            image = torch.einsum('nij,nj->ni', tensor(projection), spherical)
            values = {'B': torch.linalg.vector_norm(field, dim=-1), 'B_los': image[:, 2],
                      'B_transverse': torch.linalg.vector_norm(image[:, :2], dim=-1),
                      'azimuth': torch.rad2deg(torch.atan2(-image[:, 0], image[:, 1]))}
        for key, value in values.items():
            output[key][selected] = value.detach().cpu().numpy()
    output['azimuth'] = (output['azimuth'] + np.rad2deg(target_roll-native_roll)) % 180
    valid &= np.logical_and.reduce([np.isfinite(values) for values in output.values()])
    return SimpleNamespace(arrays={key: value.reshape(shape) for key, value in output.items()},
                           mask=valid.reshape(shape), metadata={'coordinates': {'values': coords.reshape(*shape, 4),
                            'order': 'xyzt'}, 'source_snapshot': inversion['snapshot_path']})


def read_hinode_inversion(path, *, observation=None, device='cpu', batch_size=BATCH_SIZE, shift=(0., 0.)):
    """Load native magnetic maps and exact sampling geometry from a Hinode .pme.

    An existing FITS directory may point back to its saved snapshot/provenance;
    its nominal display WCS never supplies replacement model coordinates.
    """
    from spinnme import load
    from _common import _world

    path = _path(path)
    if path.is_dir():
        with fits.open(path/'B.fits', memmap=False) as hdus:
            metadata = json.loads(hdus['PROVENANCE'].data['JSON'][0])
        observation = observation or metadata.get('observation_id')
        candidates = [path.parent/'inversion.pme']
        if metadata.get('snapshot_path'):
            candidates.insert(0, Path(metadata['snapshot_path']))
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    with load(path, device=device) as model:
        if observation is None:
            candidates = [row for row in model.observations if row.get('role') == 'train'] or model.observations
            preferred = [row for row in candidates if row.get('line_id') == 'fe-i-6302' or
                         row.get('instrument_id') == 'HINODE_6302']
            observation = (preferred or candidates)[0]['id']
        row = model.grid(observation)
        # Hinode was fitted in its original frame: preserve its native prediction.
        result = model.evaluate(row, instrument_id=row['instrument_id'], batch_size=batch_size,
                                parameters=['B', 'B_los', 'B_transverse', 'azimuth'])
        # The alignment corrects display positions and HMI sampling geometry only.
        # Slit timing and observer geometry remain unchanged.
        if row.get('slit_pixel_world') is not None:
            coefficients = np.array(row['slit_pixel_world'], dtype=float, copy=True)
            coefficients[0] += shift[0]
            coefficients[1] += shift[1]
            row = {**row, 'slit_pixel_world': coefficients.tolist()}
        elif any(shift):
            from astropy.wcs import WCS
            header = fits.Header(row['header'])
            wcs = WCS(header).celestial
            for axis, offset in enumerate(shift):
                header[f'CRVAL{axis+1}'] += (offset*u.arcsec).to_value(wcs.wcs.cunit[axis])
            row = {**row, 'header': dict(header)}
        geometry = model.geometry(row, instrument_id=row['instrument_id'])
        shape = result.mask.shape
        if row.get('slit_pixel_world') is not None:
            y_pixel, _ = np.indices(shape)
            x0, y0, dx, dy = np.asarray(row['slit_pixel_world'])
            x, y = x0+dx*y_pixel, y0+dy*y_pixel
        else:
            x, y = _world(fits.Header(row['header']), shape)
        coords = geometry['coords'].reshape(*shape, 4)
        times = (Time(model.ref_time) + coords[..., 3]*model.seconds_per_dt*u.s).unix
        roll = geometry['image_p_angle'].reshape(shape)
        arrays = {key: result.arrays['B_los' if key == 'apparent_B_los' else key] for key in PARAMETERS}
        arrays['azimuth'] = (arrays['azimuth']-np.rad2deg(roll)) % 180
        print(f"SPINNME Hinode: {path}; observation={row['id']}; native shape={shape}", flush=True)
        return {**arrays, 'x': x, 'y': y, 'times': times, 'mask': result.mask.copy(),
                'coords': coords, 'image_p_angle': roll,
                'cartesian_to_spherical_transform': geometry['cartesian_to_spherical_transform'].reshape(*shape, 3, 3),
                'rtp_to_img_transform': geometry['rtp_to_img_transform'].reshape(*shape, 3, 3),
                'Rs_per_ds': model.Rs_per_ds, 'seconds_per_dt': model.seconds_per_dt,
                'ref_time': model.ref_time, 'snapshot_path': str(path)}


def _figures(output, arrays, bmax=3000., dpi=400, *, bounds=None):
    """Four map columns, dedicated colorbars and cumulative pixel distributions."""
    import cmasher as cmr
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    from matplotlib.ticker import MaxNLocator
    x, y = arrays['x_arcsec'], arrays['y_arcsec']
    sources = SOURCES
    grids = {source: (arrays.get(source+'_x_arcsec', x), arrays.get(source+'_y_arcsec', y))
             for source in sources}
    xmin = min(np.nanmin(gx) for gx, _ in grids.values())
    ymin = min(np.nanmin(gy) for _, gy in grids.values())
    ymax = max(np.nanmax(gy) for _, gy in grids.values())
    xmax = PLOT_X_MAX_ARCSEC
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError('Plot bounds do not overlap the selected grids; set --hpc-range explicitly.')
    panel_height = 1.65*(ymax-ymin)/(xmax-xmin)
    fig = plt.figure(figsize=(14., 3*max(panel_height, 1.65)+1.1), layout='constrained')
    gs = fig.add_gridspec(3, 6, width_ratios=(1, 1, 1, 1, .045, 1.35))
    scales = ((SymLogNorm(linthresh=1., vmin=-bmax, vmax=bmax), 'RdBu_r', r'$B_\mathrm{LOS}$ [G]'),
              (LogNorm(1., bmax, clip=True), 'viridis', r'$B_\mathrm{trv}$ [G]'),
              (Normalize(0, 180), cmr.infinity, r'Azimuth mod $180^\circ$ [$^\circ$]'))
    titles = ('HMI reference 720 s', 'SPINNME HMI', 'Hinode MERLIN', 'SPINNME Hinode')
    for row, (parameter, (norm, cmap, label)) in enumerate(zip(PLOT_PARAMETERS, scales)):
        for column, source in enumerate(sources):
            ax = fig.add_subplot(gs[row, column])
            own_x, own_y = grids[source]
            image = ax.pcolormesh(own_x, own_y, np.ma.masked_invalid(arrays[source+'_'+parameter]),
                                  shading='nearest', norm=norm, cmap=cmap, rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(top=False, right=False, labelbottom=row == 2, labelleft=column == 0,
                           direction='out', length=3, pad=2, labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune='both'))
            if row == 2:
                ax.set_xlabel('Solar X [arcsec]', fontsize=9)
            if column == 0:
                ax.set_ylabel('Solar Y [arcsec]', fontsize=9)
            if row == 0:
                ax.set_title(titles[column], fontsize=10, pad=9)
        cax = fig.add_subplot(gs[row, 4])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(label, fontsize=9)
        bar.ax.tick_params(labelsize=8)
        if parameter == 'azimuth':
            bar.set_ticks([0, 45, 90, 135, 180])
        distribution = fig.add_subplot(gs[row, 5])
        _plot_distribution(distribution, arrays, grids, parameter, norm, label,
                           (xmin, xmax, ymin, ymax), titles)
    save_figure(fig, output/'comparison', ('pdf', 'png'), dpi)
    plt.close(fig)


def _plot_distribution(distribution, arrays, grids, parameter, norm, label, bounds, titles):
    """Exact ECDF on visible native pixels; field strengths use magnitudes."""
    from matplotlib.ticker import PercentFormatter
    xmin, xmax, ymin, ymax = bounds
    sources = SOURCES
    limits = [norm.vmin, norm.vmax]
    if parameter in ('apparent_B_los', 'B_transverse'):
        limits[0] = 0.
    for source, title, color, style in zip(sources, titles,
                                           ('#0072B2', '#009E73', '#D55E00', '#CC79A7'),
                                           ('-', ':', '--', '-.')):
        own_x, own_y = grids[source]
        visible = np.isfinite(own_x) & np.isfinite(own_y)
        visible &= (own_x >= xmin) & (own_x <= xmax) & (own_y >= ymin) & (own_y <= ymax)
        visible &= np.isfinite(arrays[source+'_'+parameter])
        values = arrays[source+'_'+parameter][visible]
        if parameter == 'apparent_B_los':
            values = np.abs(values)
        weights = np.ones(values.shape)
        if parameter == 'azimuth':
            strength = arrays[source+'_B'][visible]
            weights = np.where(np.isfinite(strength), np.clip((strength-10.)/90., 0., 1.), 0.)
        # Exact weighted ECDF: retain tails outside the map color limits and
        # combine ties so each step includes all pixels at that value.
        keep = weights > 0
        values, weights = values[keep], weights[keep]
        if not values.size:
            continue
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        unique, first = np.unique(values, return_index=True)
        cumulative = np.cumsum(np.add.reduceat(weights, first))
        cumulative /= cumulative[-1]
        limits = [min(limits[0], unique[0]), max(limits[1], unique[-1])]
        distribution.step(np.r_[unique[0], unique], np.r_[0., cumulative], where='post',
                          lw=1.5, color=color, linestyle=style, label=title)
    # Extend each ECDF to the common bounds, making the 0/100% tails explicit.
    for line in distribution.lines:
        values, fractions = line.get_data()
        line.set_data(np.r_[limits[0], values, limits[1]], np.r_[0., fractions, 1.])
    if parameter in ('apparent_B_los', 'B_transverse'):
        limits[0] = 0.
        distribution.set_xscale('symlog', linthresh=1.)
        exponents = np.arange(0, int(np.floor(np.log10(limits[1]))) + 1)
        ticks = np.r_[0., 10. ** exponents]
        distribution.set_xticks(ticks, labels=['0'] + [rf'$10^{{{power}}}$' for power in exponents])
    distribution_label = r'$|B_\mathrm{LOS}|$ [G]' if parameter == 'apparent_B_los' else label
    distribution.set(xlim=limits, ylim=(0, 1), xlabel=distribution_label)
    distribution.xaxis.label.set_size(9)
    distribution.set_ylabel('Weighted pixels ≤ value' if parameter == 'azimuth'
                            else 'Pixels ≤ value', fontsize=9)
    distribution.tick_params(labelsize=8, direction='out', length=3, pad=2)
    distribution.set_yticks([0, .5, 1])
    distribution.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    distribution.grid(axis='y', color='.9', linewidth=.6)
    distribution.spines[['top', 'right']].set_visible(False)
    if parameter == 'azimuth':
        distribution.set_xticks([0, 90, 180])
    if parameter == PLOT_PARAMETERS[0]:
        distribution.set_title('Cumulative distributions', fontsize=10, pad=9)
        distribution.legend(fontsize=7, frameon=False, loc='lower right')


def aligned_crop(source, begin, finish, bounds, shift):
    """Select a display grid without changing the source or physical xyzt."""
    displayed = {**source, 'x': source['x']+shift[0], 'y': source['y']+shift[1]}
    return native_crop(displayed, begin, finish, bounds)


def reference_rolls(records):
    """Compute each HMI image basis once, in record order."""
    from sunpy.map import Map
    from spinnme.data.util import image_p_angle
    return np.array([image_p_angle(Map(np.zeros((1, 1)), row['header'])) for row in records])


def matched_roll(times, records, rolls):
    """Use the nearest HMI basis without changing native model times or masks."""
    indices, _, _ = pair_times(times, records)
    return rolls[indices]


def reference_maps(grid, records):
    """Reproject HMI onto the common Hinode positions at nearest slit epochs."""
    indices, _, time_valid = pair_times(grid['times'], records)
    mask = grid['mask'] & time_valid
    if not mask.any():
        raise ValueError('No finite Hinode pixels have an HMI reference within 360 seconds.')
    hmi = {key: np.full(mask.shape, np.nan) for key in PARAMETERS}
    selected = np.unique(indices[mask])
    print(f'Reprojecting {len(selected)} HMI maps onto the common Hinode grid...', flush=True)
    for index in selected:
        record = records[index]
        local = mask & (indices == index)
        x, y = grid['x'][local], grid['y'][local]
        hmi['azimuth'][local] = _sample_image(record['azimuth'], x, y, nearest=True) % 180
        for parameter in PARAMETERS[:-1]:
            hmi[parameter][local] = _sample_image(
                record['field'], x, y, limits=(0, np.inf),
                magnetic=(record['inclination'], parameter, 1))
    mask &= np.logical_and.reduce([np.isfinite(hmi[key]) for key in PARAMETERS])
    if not mask.any():
        raise ValueError('No finite common Hinode/HMI samples remain.')
    return hmi, mask


def add_map(arrays, name, grid, values, mask):
    """Attach each map to its own explicit display coordinates and mask."""
    arrays[name+'_x_arcsec'], arrays[name+'_y_arcsec'] = grid['x'], grid['y']
    arrays.update({name+'_'+key: np.where(mask, values[key], np.nan) for key in PARAMETERS})


def compare(output, *, snapshot=SNAPSHOT, hinode_result=HINODE_RESULT, reference=HMI_REFERENCE, hinode=HINODE_FILE,
            start=START, end=END, time_scale=TIME_SCALE, bounds=None, shift_x=0., shift_y=0.,
            hinode_fit_status=None, hinode_observation=None, device=DEVICE, batch_size=BATCH_SIZE, bmax=3000., dpi=400):
    """Load → align Hinode → derive common geometry → sample/reproject → plot."""
    from spinnme import load

    output = _path(output)
    if not np.isfinite(bmax) or bmax <= 1 or dpi < 1:
        raise ValueError('bmax must exceed 1 G and dpi must be positive.')
    if not np.isfinite([shift_x, shift_y]).all():
        raise ValueError('Hinode shifts must be finite arcsecond values.')
    begin, finish = time_window(start, end, time_scale)
    shift = (shift_x, shift_y)

    # 1. Load immutable source geometry and reference epochs. Keep neighboring
    # records outside the requested window for pairing pixels near its edges.
    print('Loading Hinode coordinates and HMI references...', flush=True)
    source = merlin({**HINODE, 'file': str(hinode), 'accepted_fit_status': hinode_fit_status})
    native = read_hinode_inversion(hinode_result, observation=hinode_observation,
                                  device=device, batch_size=batch_size, shift=shift)
    records = hmi_records(reference)
    rolls = reference_rolls(records)

    # 2. Crop the already aligned common grid. MERLIN retains its native pixels.
    merlin_grid = aligned_crop(source, begin, finish, bounds, shift)
    grid = native_crop(native, begin, finish, bounds)

    # 3. Reproject HMI onto the same shifted Hinode grid; MERLIN is displayed
    # in that frame without interpolating its native inversion values.
    hmi, reference_mask = reference_maps(grid, records)
    arrays = {'x_arcsec': grid['x'], 'y_arcsec': grid['y']}
    add_map(arrays, 'hmi', grid, hmi, reference_mask)
    hinode = _magnetic(merlin_grid['B'], merlin_grid['inclination'], HINODE['los_sign'], merlin_grid['f'])
    hinode['azimuth'] = (merlin_grid['azimuth'] + HINODE['azimuth_rotation_deg'] +
                         np.rad2deg(matched_roll(merlin_grid['times'], records, rolls))) % 180
    add_map(arrays, 'hinode', merlin_grid, hinode, merlin_grid['mask'])
    roll = matched_roll(grid['times'], records, rolls)
    hinode_values = {key: grid[key] for key in PARAMETERS}
    hinode_values['azimuth'] = (hinode_values['azimuth'] + np.rad2deg(roll)) % 180
    add_map(arrays, 'spinnme_hinode', grid, hinode_values, grid['mask'])

    # 4. Evaluate HMI at the shifted physical positions and native slit times.
    # Hinode values remain evaluated in the unshifted frame in which it was fitted.
    print(f"Sampling SPINNME HMI at {grid['mask'].sum()} aligned Hinode xyzt coordinates...", flush=True)
    with load(_path(snapshot), device=device) as model:
        prediction = sample_snapshot(model, grid, batch_size, image_roll=roll)
    values = {key: prediction.arrays['B_los' if key == 'apparent_B_los' else key] for key in PARAMETERS}
    add_map(arrays, 'spinnme', grid, values, prediction.mask)

    # 5. Render maps and distributions from the same display bounds.
    output.mkdir(parents=True, exist_ok=True)
    print('Saving comparison plot...', flush=True)
    _figures(output, arrays, bmax, dpi, bounds=bounds)
    return output/'comparison.png'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(SNAPSHOT))
    parser.add_argument('--hinode-result', type=Path, default=Path(HINODE_RESULT),
                        help='Hinode inversion.pme, or FITS directory pointing to its source snapshot.')
    parser.add_argument('--hinode-observation', help='Saved Hinode observation ID; defaults to the first training Fe I 6302 grid.')
    parser.add_argument('--reference', type=Path, default=Path(HMI_REFERENCE))
    parser.add_argument('--hinode', type=Path, default=Path(HINODE_FILE))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', default=START)
    parser.add_argument('--end', default=END)
    parser.add_argument('--time-scale', choices=('utc', 'tai'), default=TIME_SCALE)
    parser.add_argument('--hpc-range', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    parser.add_argument('--shift-x', type=float, default=0., help='Correct Hinode solar X [arcsec] before deriving common model coordinates.')
    parser.add_argument('--shift-y', type=float, default=0., help='Correct Hinode solar Y [arcsec] before deriving common model coordinates.')
    parser.add_argument('--hinode-fit-status', type=int, nargs='+', choices=range(1, 7),
                        help='Optional native fit-status filter; default uses finite physical values.')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default=DEVICE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--bmax', type=float, default=3000., help='Shared magnetic field upper limit [G], matching the cadence plate.')
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args(argv)
    try:
        print(compare(args.output, snapshot=args.input, hinode_result=args.hinode_result, reference=args.reference, hinode=args.hinode,
                      start=args.start, end=args.end, time_scale=args.time_scale, bounds=args.hpc_range,
                      shift_x=args.shift_x, shift_y=args.shift_y,
                      hinode_fit_status=args.hinode_fit_status, hinode_observation=args.hinode_observation,
                      device=args.device, batch_size=args.batch_size, bmax=args.bmax, dpi=args.dpi))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f'comparison: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
