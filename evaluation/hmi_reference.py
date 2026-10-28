"""Paper figures from one SPINNME savestate and one epoch of HMI inversion FITS.

Adapted from pinn-me/pme/evaluation/spherical/load_ref_map.py and
load_ref_comparison.py. Run from the spinnme-evaluation project root.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import cmasher as cmr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from astropy import units as u
from astropy.time import Time
from sunpy.map import Map

from spinnme.plotting import magnetic_norm, plot_comparison, plot_maps, plot_scatter

from _common import (_header, _model_time, _record_key, _sample_image, _world,
                     comparison, save_figure, write_json)

LABELS = {'B': r'$|B|$', 'inclination': r'$\gamma$', 'azimuth': r'$\chi\;\mathrm{mod}\;180^\circ$',
          'Br': r'$B_r$', 'Btheta': r'$B_\theta$', 'Bphi': r'$B_\phi$',
          'vr': r'$v_r$', 'vtheta': r'$v_\theta$', 'vphi': r'$v_\phi$', 'v_los': r'$v_\mathrm{LOS}$'}
UNITS = {key: 'G' if key.startswith('B') else 'deg' if key in ('inclination', 'azimuth') else 'km/s'
         for key in LABELS}
PARAMETERS = ['B', 'inclination', 'azimuth', 'Br', 'Btheta', 'Bphi', 'vr', 'vtheta', 'vphi', 'v_los']
HMI_UNITS = {'field': 'G', 'inclination': 'deg', 'azimuth': 'deg', 'vlos_mag': 'cm/s'}


def notice(message, notes=None):
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    if notes is not None:
        notes.append(message)


def reference_record(directory, time=None, notes=None):
    """Select a real field epoch, then require the other three matching segments."""
    directory = Path(directory)
    fields = [(path, _header(path)[0]) for path in sorted(directory.glob('*.field.fits'))]
    if not fields:
        raise FileNotFoundError(f'No *.field.fits in {directory}')
    fields.sort(key=lambda row: _model_time(row[1]).unix)
    selected = fields[len(fields)//2] if time is None else min(
        fields, key=lambda row: abs(_model_time(row[1]).unix-Time(time).unix))
    key = _record_key(selected[1])
    result = {'field': selected[0]}
    for segment in ('inclination', 'azimuth', 'vlos_mag', 'disambig', 'conf_disambig'):
        candidates = list(directory.glob(f'*.{segment}.fits'))
        if segment in ('disambig', 'conf_disambig'):
            candidates += list((directory/'disambiguation').glob(f'*.{segment}.fits'))
        matches = [path for path in sorted(candidates)
                   if _record_key(_header(path)[0]) == key]
        if not matches:
            # JSOC exports sometimes omit T_REC or give segment-specific DATE-OBS.
            sibling = selected[0].with_name(selected[0].name.replace('.field.fits', f'.{segment}.fits'))
            if sibling.is_file():
                matches = [sibling]
                notice(f'Pairing {segment} by archive filename; record metadata differ.', notes)
        if not matches:
            if segment in ('disambig', 'conf_disambig'):
                continue
            raise ValueError(f'Expected one {segment} segment for T_REC={key}; found none.')
        if len(matches) > 1:
            notice(f'Multiple {segment} files for T_REC={key}; using {matches[0]}.', notes)
        result[segment] = matches[0]
    if sum(_record_key(header) == key for _, header in fields) != 1:
        notice(f'Multiple field files for T_REC={key}; using {selected[0]}.', notes)
    return result


def reference_grid(files, snapshot, hpc_range=None, resolution=1024, notes=None):
    """Crop before constructing an optional smaller common comparison grid."""
    header, shape = _header(files['field'])
    epoch = _model_time(header)
    missing_velocity = [key for key in ('OBS_VR', 'OBS_VW', 'OBS_VN') if key not in header]
    if missing_velocity:
        notice(f'Missing {", ".join(missing_velocity)}; unavailable observer velocity components default to zero.', notes)
    for segment, path in files.items():
        other, other_shape = _header(path)
        if abs((_model_time(other)-epoch).to_value(u.s)) > 1:
            notice(f'{segment} T_OBS differs from field T_OBS; using the field epoch for the model.', notes)
        if other_shape != shape:
            notice(f'{segment} has a different shape; sampling its WCS onto the common field grid.', notes)
        for key in ('DSUN_OBS', 'CRLN_OBS', 'CRLT_OBS'):
            if key in header and key in other and not np.isclose(float(other[key]), float(header[key]), rtol=1e-6, atol=1e-4):
                notice(f'{segment} {key} differs from field metadata; using the field observer geometry.', notes)
        if 'QUALITY' in other and int(other['QUALITY']) != 0:
            notice(f'Nonzero HMI QUALITY in {path}: {other["QUALITY"]}; retaining available finite pixels.', notes)
    from spinnme.data.hmi_time import hmi_coordinate_header
    header = hmi_coordinate_header(header)
    bounds = None
    if hpc_range is not None:
        xmin, xmax, ymin, ymax = hpc_range
        bounds = (xmin, ymin, xmax, ymax)
    if resolution < 0 or resolution == 1:
        raise ValueError('--resolution must be zero (native) or at least two.')
    observation = snapshot.grid({'id': 'hmi-reference', 'instrument_id': 'HMI',
                                'shape': list(shape), 'header': dict(header),
                                'time': {'model_time': epoch.utc.isot}}, bounds=bounds)
    if resolution and max(observation['shape']) > resolution:
        dimensions = np.maximum(2, np.rint(np.array(observation['shape'])*resolution/max(observation['shape'])).astype(int))
        observation = snapshot.grid(observation, shape=tuple(dimensions))
    grid = Map(np.broadcast_to(np.zeros((1, 1), dtype=np.uint8), observation['shape']), observation['header'])
    return grid, epoch, observation


def sample_reference(files, grid, notes=None, *, azimuth_disambiguated=False, disambiguation=None):
    x, y = _world(grid.fits_header, grid.data.shape)
    arrays = {}
    targets = {'field': ('B', u.G), 'inclination': ('inclination', u.deg),
               'azimuth': ('azimuth', u.deg), 'vlos_mag': ('v_los', u.km/u.s)}
    for segment, (name, unit) in targets.items():
        header, _ = _header(files[segment])
        declared = str(header.get('BUNIT') or '').strip()
        if declared.lower() in ('', 'none', 'undefined', 'unknown', 'n/a', '?'):
            declared = HMI_UNITS[segment]
            notice(f'Missing/unspecified BUNIT in {files[segment]}; assuming standard HMI {segment} units: {declared}.', notes)
        declared = {'gauss': 'G', 'degrees': 'deg', 'degree': 'deg', 'radians': 'rad'}.get(declared.lower(), declared)
        factor = u.Unit(declared).to(unit)
        arrays[name] = _sample_image(files[segment], x, y, nearest=True)*factor
    arrays['B'][arrays['B'] < 0] = np.nan
    arrays['inclination'][(arrays['inclination'] < 0) | (arrays['inclination'] > 180)] = np.nan
    header, _ = _header(files['azimuth'])
    series = str(header.get('SERIES', header.get('JSOC_SER', ''))).lower()+' '+files['azimuth'].name.lower()
    if azimuth_disambiguated or 'hmi.sharp_' in series or 'hmi.sharp.' in series:
        arrays['azimuth_brtp'] = arrays['azimuth'].copy()
        status = {'status': 'already_applied', 'source': 'explicit flag or SHARP series'}
    elif all(key in files for key in ('disambig', 'conf_disambig')):
        bitmap = _sample_image(files['disambig'], x, y, nearest=True)
        confidence = _sample_image(files['conf_disambig'], x, y, nearest=True)
        arrays['azimuth_brtp'], flipped = disambiguate_azimuth(arrays['azimuth'], bitmap, confidence)
        status = {'status': 'applied', 'strong_intermediate_bit': 0, 'weak_bit': 2,
                  'weak_method': 'radial acute angle', 'flipped_pixels': int(flipped.sum()),
                  'valid_pixels': int(np.isfinite(arrays['azimuth_brtp']).sum())}
    else:
        arrays['azimuth_brtp'] = np.full_like(arrays['azimuth'], np.nan)
        status = {'status': 'missing_maps'}
        notice('Missing disambig/conf_disambig: skipping signed Brtp comparisons. Download both matching HMI segments; '
               'use --azimuth-disambiguated only for azimuth maps already corrected upstream.', notes)
    if disambiguation is not None:
        disambiguation.update(status)
    return arrays


def disambiguate_azimuth(azimuth, bitmap, confidence):
    """HMI full-disk convention, Hoeksema et al. 2014, Appendix A.4/Table 7.

    Bit 0: minimum-energy/smoothed solution at confidence 90/60.
    Bit 2: radial-acute solution at confidence 50. Confidence 0 is unresolved.
    Test the selected bit alone; higher bits must not trigger a flip.
    """
    valid = np.isfinite(azimuth) & np.isfinite(bitmap) & (bitmap >= 0) & (bitmap <= 255)
    valid &= (bitmap == np.floor(bitmap)) & np.isin(confidence, (50, 60, 90))
    bits = np.where(valid, bitmap, 0).astype(np.uint8)
    bit_index = np.where(confidence == 50, 2, 0)
    flipped = valid & (((bits >> bit_index) & 1) != 0)
    corrected = np.where(valid, (azimuth % 180+180*flipped) % 360, np.nan)
    return corrected, flipped


def magnetic_vector(strength, inclination, azimuth):
    """HMI CCD convention: azimuth CCW from +y, LOS toward the observer."""
    inc, azi = np.deg2rad(inclination), np.deg2rad(azimuth)
    return np.stack([-strength*np.sin(inc)*np.sin(azi),
                     strength*np.sin(inc)*np.cos(azi), strength*np.cos(inc)], axis=-1)


def transform_reference(reference, grid, snapshot, observation, batch_size=8192):
    """Use exactly the same finite-distance geometry as Snapshot.evaluate.

    HMI vlos_mag is observer-frame Doppler velocity. Snapshot v_los already
    contains solar rotation, so remove only observer motion from the reference.
    Brtp uses the pipeline disambiguation applied in sample_reference(), never
    a model-dependent branch choice. The angle plot remains modulo 180 degrees.
    """
    shape = grid.data.shape
    b_img = magnetic_vector(reference['B'], reference['inclination'], reference.pop('azimuth_brtp')).reshape(-1, 3)
    n = len(b_img)
    b_rtp = np.full((n, 3), np.nan)
    v_obs = np.full(n, np.nan)
    for start in range(0, n, batch_size):
        stop = min(n, start+batch_size)
        geometry = snapshot.geometry(observation, indices=np.arange(start, stop))
        b_rtp[start:stop] = np.einsum('...ji,...j->...i', geometry['rtp_to_img_transform'], b_img[start:stop])
        v_obs[start:stop] = geometry['v_obs_los'].reshape(-1)/1000
    reference.update({key: b_rtp[:, i].reshape(shape) for i, key in enumerate(('Br', 'Btheta', 'Bphi'))})
    reference['v_los'] = reference['v_los']-v_obs.reshape(shape)
    reference['azimuth'] %= 180
    return v_obs.reshape(shape)


def map_norm(name, predicted, reference, mask):
    if name in ('inclination', 'azimuth'):
        return Normalize(0, 180), cmr.infinity if name == 'azimuth' else 'PiYG'
    values = predicted[mask] if reference is None else np.r_[predicted[mask], reference[mask]]
    norm = magnetic_norm(name, values)
    if norm is not None:
        return norm, 'viridis' if name == 'B' else 'RdBu_r'
    bound = max(float(np.percentile(np.abs(values), 99)), 1e-3)
    return Normalize(-bound, bound), 'RdBu_r'


def comparison_figure(names, prediction, reference, masks, header, output, title, dpi):
    options = {name: map_norm(name, prediction[name], reference[name], masks[name]) for name in names}
    figure = plot_comparison(prediction, reference, names, header=header, masks=masks,
        labels=LABELS, units=UNITS, norms={key: value[0] for key, value in options.items()},
        cmaps={key: value[1] for key, value in options.items()}, periods={'azimuth': 180},
        source_labels=('HMI Pipeline', 'SPINNME'), title=title, difference_floor=.01)
    save_figure(figure, output, ('pdf', 'png'), dpi)
    plt.close(figure)


def velocity_figure(prediction, mask, header, output, title, dpi):
    names = ('vr', 'vtheta', 'vphi')
    options = {name: map_norm(name, prediction[name], None, mask & np.isfinite(prediction[name])) for name in names}
    figure = plot_maps(prediction, names, header=header, masks=mask, labels=LABELS, units=UNITS,
        norms={key: value[0] for key, value in options.items()},
        cmaps={key: value[1] for key, value in options.items()}, source_label='SPINNME',
        title=title+'\nCarrington-frame velocity; HMI has no vector-velocity reference')
    save_figure(figure, output, ('pdf', 'png'), dpi)
    plt.close(figure)


def scatter_figure(prediction, reference, masks, output, dpi):
    names = [name for name in ('B', 'inclination', 'azimuth', 'Br', 'Btheta', 'Bphi', 'v_los') if name in masks]
    figure = plot_scatter(prediction, reference, names, masks=masks, labels=LABELS, units=UNITS,
        periods={'azimuth': 180}, source_labels=('HMI', 'SPINNME'),
        title='Paired pixels · azimuth shown on the nearest 180° branch')
    save_figure(figure, output, ('pdf', 'png'), dpi)
    plt.close(figure)


def evaluate(input_path, reference_directory, output, *, time=None, hpc_range=None,
             resolution=1024, device='auto', batch_size=8192, dpi=400, azimuth_disambiguated=False):
    from spinnme import load
    from spinnme.exports import EvaluationResult
    input_path, output = Path(input_path), Path(output)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    notes = []
    disambiguation = {}
    files = reference_record(reference_directory, time, notes)
    with load(input_path, device=device) as snapshot:
        grid, epoch, observation = reference_grid(files, snapshot, hpc_range, resolution, notes)
        reference = sample_reference(files, grid, notes, azimuth_disambiguated=azimuth_disambiguated,
                                     disambiguation=disambiguation)
        print(f'HMI reference: {files["field"]}\nModel epoch: {epoch.utc.isot} UTC; grid: {grid.data.shape}', flush=True)
        result = snapshot.evaluate(observation, instrument_id='HMI', time=epoch.utc.isot,
                                   parameters=PARAMETERS, batch_size=batch_size)
        v_obs = transform_reference(reference, grid, snapshot, observation, batch_size)
    prediction = {name: values/1000 if name in ('vr', 'vtheta', 'vphi', 'v_los') else values
                  for name, values in result.arrays.items()}
    masks, residuals, metrics = {}, {}, []
    for name in ('B', 'inclination', 'azimuth', 'Br', 'Btheta', 'Bphi', 'v_los'):
        mask = result.mask & np.isfinite(prediction[name]) & np.isfinite(reference[name])
        if name in ('inclination', 'azimuth'):
            mask &= np.isfinite(reference['B']) & (reference['B'] > 0) & (prediction['B'] > 0)
        if not mask.any():
            if name in ('Br', 'Btheta', 'Bphi'):
                notice(f'No resolved reference pixels for {name}; omitting its metrics and comparison plot.', notes)
                continue
            raise ValueError(f'No common finite pixels for {name}.')
        metric, difference = comparison(prediction[name], reference[name], mask,
                                        period=180 if name == 'azimuth' else None)
        masks[name], residuals[name] = mask, difference
        metrics.append({'parameter': name, 'unit': UNITS[name], **metric})
    output.mkdir(parents=True, exist_ok=True)
    result.write_npz(output/'spinnme.npz', overwrite=True)
    reference_mask = result.mask & np.logical_and.reduce([np.isfinite(reference[key]) for key in ('B', 'inclination', 'azimuth')])
    EvaluationResult(reference, {name: UNITS[name] for name in reference}, grid.fits_header, reference_mask,
                     {'model_time': epoch.utc.isot, 'reference': 'HMI pipeline',
                      'velocity': 'vlos_mag minus observer LOS velocity; includes solar rotation'}).write_npz(
                          output/'hmi.npz', overwrite=True)
    np.savez_compressed(output/'comparison_arrays.npz',
                        **{f'mask_{key}': value for key, value in masks.items()},
                        **{f'residual_{key}': value for key, value in residuals.items()}, observer_vlos_km_s=v_obs)
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    write_json(output/'comparison.json', {'snapshot': str(input_path.resolve()),
        'references': {name: str(path.resolve()) for name, path in files.items()},
        'requested_time_utc': time, 'model_time_utc': epoch.utc.isot, 'shape_yx': list(grid.data.shape),
        'hpc_range_xmin_xmax_ymin_ymax_arcsec': hpc_range, 'resolution_max_dimension': resolution,
        'sampling': 'Nearest native pixels on a common WCS; no PSF matching or spatial averaging.',
        'magnetic_branch': 'HMI pipeline disambiguation and original SPINNME branch; no model-dependent flips.',
        'disambiguation': disambiguation,
        'angular_residual': 'Azimuth modulo 180 degrees; inclination is a bounded angle, never modulo 180.',
        'velocity': 'v_los is positive redshift, observer motion removed, solar rotation retained; no offset fitting.',
        'vector_velocity': 'vr/vtheta/vphi are Carrington-frame model outputs; no vector-velocity HMI reference.',
        'reference_role': 'Operational pipeline inversion, not ground truth.',
        'warnings': notes, 'metrics': metrics})
    title = epoch.utc.isot+' UTC'
    comparison_figure(('B', 'inclination', 'azimuth'), prediction, reference, masks,
                      grid.fits_header, output/'inversion_parameters', title, dpi)
    vector_names = [name for name in ('Br', 'Btheta', 'Bphi') if name in masks]
    if vector_names:
        comparison_figure(vector_names, prediction, reference, masks,
                          grid.fits_header, output/'magnetic_vector', title+' · HMI pipeline disambiguation', dpi)
    else:
        for extension in ('pdf', 'png'):
            (output/f'magnetic_vector.{extension}').unlink(missing_ok=True)
    comparison_figure(('v_los',), prediction, reference, masks,
                      grid.fits_header, output/'los_velocity', title+' · observer motion removed', dpi)
    velocity_figure(prediction, result.mask, grid.fits_header, output/'velocity_vector', title, dpi)
    scatter_figure(prediction, reference, masks, output/'scatter', dpi)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='SPINNME inversion.pme savestate')
    parser.add_argument('--reference', required=True, type=Path,
                        help='Directory containing HMI field, inclination, azimuth and vlos_mag FITS')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--time', help='UTC epoch: select nearest HMI field record (default: middle record)')
    parser.add_argument('--hpc-range', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Optional subframe in helioprojective arcsec, as in the original archive script')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Maximum output dimension after crop (default: 1024); 0 retains native pixels')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--dpi', type=int, default=400)
    parser.add_argument('--azimuth-disambiguated', action='store_true',
                        help='Input azimuth already has disambiguation applied; do not apply DISAMBIG again')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.dpi < 1:
        parser.error('--batch-size and --dpi must be positive.')
    try:
        print(evaluate(args.input, args.reference, args.output, time=args.time, hpc_range=args.hpc_range,
                       resolution=args.resolution, device=args.device, batch_size=args.batch_size, dpi=args.dpi,
                       azimuth_disambiguated=args.azimuth_disambiguated))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'HMI reference comparison: {exc}\n')


if __name__ == '__main__':
    main()
