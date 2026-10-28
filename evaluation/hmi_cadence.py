"""Plot HMI cadence inversions against the same HMI90 pipeline maps and epochs.

Inputs are the two public .pme snapshots and archive FITS.
The selected native 90-second and/or 720-second products are evaluated on common epochs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.map import Map

from spinnme import load
from spinnme.exports import EvaluationResult, read_npz

from _common import (_header, _model_time, _record_key, _series, _sample_image, _world,
                     comparison, save_figure, write_json)


# Analysis choices for the March 23 two-hour HMI cadence comparison.
PARAMETERS = ('B', 'B_los', 'B_transverse')
SHAPE = (512, 512)
BOUNDS_HPC_ARCSEC = None  # xmin,ymin,xmax,ymax; None selects the full disk.
TIME_STRIDE = 8
LOS_SIGN = 1
ALLOWED_QUALITY_BITS = 0
ACCEPTED_CONFIDENCE_VALUES = (0, 1, 2, 3)
DEVICE = None
BATCH_SIZE = 8192


def _csv(path, rows):
    if not rows:
        raise ValueError('No HMI cadence evaluation rows were produced.')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _runs(directory, selected=None):
    rows = [
        {'run_id': 'hmi90', 'cadence_s': 90,
         'snapshot': directory/'hmi_fd_20240323_2h_90s_v01/inversion.pme'},
        {'run_id': 'hmi720', 'cadence_s': 720,
         'snapshot': directory/'hmi_fd_20240323_2h_720s_v01/inversion.pme'},
    ]
    if selected is not None:
        unknown = set(selected) - {row['run_id'] for row in rows}
        if unknown or not selected:
            raise ValueError('Select hmi90 and/or hmi720.')
        rows = [row for row in rows if row['run_id'] in selected]
    for row in rows:
        if not row['snapshot'].is_file():
            raise FileNotFoundError(row['snapshot'])
    return rows


def _references(directory):
    channels = {}
    for name, suffix in (('field', 'field'), ('inclination', 'inclination'), ('confidence', 'confid_map')):
        rows = {}
        for path in sorted(directory.glob(f'*.{suffix}.fits')):
            header, shape = _header(path)
            _series(header, path, 'hmi.B_90s')
            if 'QUALITY' not in header:
                raise ValueError(f'Missing HMI QUALITY: {path}')
            if int(header['QUALITY']) & ~int(ALLOWED_QUALITY_BITS):
                continue
            key = _record_key(header)
            if key in rows:
                raise ValueError(f'Duplicate HMI90 {name} record {key}.')
            rows[key] = {'path': path, 'header': header, 'shape': shape}
        channels[name] = rows
    common = set.intersection(*(set(rows) for rows in channels.values()))
    records = [{**{name: rows[key] for name, rows in channels.items()},
                'time': _model_time(channels['field'][key]['header']).utc.isot} for key in common]
    records.sort(key=lambda row: Time(row['time']).unix)
    if not records:
        raise ValueError('No quality-approved paired B_90s field/inclination/confid_map files. Download references; B_720s is not substituted.')
    return records


def _training_span(runs):
    starts, stops = [], []
    for run in runs:
        with load(run['snapshot'], device='cpu') as snapshot:
            epochs = []
            for observation in snapshot.observations:
                if observation.get('role') != 'train' or observation.get('instrument_id') != 'HMI':
                    continue
                time = Time(observation['time']['model_time']).unix
                offsets = np.asarray(observation.get('time_offsets_seconds', [0]), dtype=float)
                epochs.extend([time+offsets.min(), time+offsets.max()])
            if not epochs:
                raise ValueError(f"No saved training epochs in {run['snapshot']}.")
            starts.append(min(epochs))
            stops.append(max(epochs))
    if max(starts) > min(stops):
        raise ValueError('The selected inversions have no common training time span.')
    return max(starts), min(stops)


def _reference(record):
    header = record['field']['header'].copy()
    required = ('DSUN_OBS', 'RSUN_REF', 'CRLN_OBS', 'CRLT_OBS', 'OBS_VR', 'OBS_VW', 'OBS_VN')
    if any(key not in header for key in required):
        raise ValueError('HMI reference requires complete observer position/velocity and solar spatial WCS.')
    from spinnme.data.hmi_time import hmi_coordinate_header
    header = hmi_coordinate_header(header)
    grid = Map(np.ones(record['field']['shape'], dtype=np.uint8), header)
    if BOUNDS_HPC_ARCSEC is not None:
        bounds = np.asarray(BOUNDS_HPC_ARCSEC, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2]):
            raise ValueError('bounds_hpc_arcsec is xmin,ymin,xmax,ymax.')
        grid = grid.submap(SkyCoord(bounds[0]*u.arcsec, bounds[1]*u.arcsec, frame=grid.coordinate_frame),
                           top_right=SkyCoord(bounds[2]*u.arcsec, bounds[3]*u.arcsec, frame=grid.coordinate_frame))
    if not grid.data.size:
        raise ValueError('The requested HMI comparison region is empty.')
    if SHAPE is not None:
        shape = SHAPE
        if len(shape) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in shape):
            raise ValueError('shape must contain two integer dimensions >=2, in ny,nx order.')
        grid = grid.resample([shape[1], shape[0]]*u.pix, method='nearest')
    header = grid.fits_header
    x, y = _world(header, grid.data.shape)
    parameters = PARAMETERS
    if not set(parameters) <= {'B', 'B_los', 'B_transverse', 'inclination'}:
        raise ValueError('HMI pipeline comparisons support B, B_los, B_transverse and inclination.')
    sign = LOS_SIGN
    if sign not in (-1, 1):
        raise ValueError('Set verified HMI los_sign: +1 for inclination zero toward the observer, otherwise -1.')
    confidence = ACCEPTED_CONFIDENCE_VALUES
    if not confidence or not set(confidence) <= set(range(7)):
        raise ValueError('accepted_confidence_values must select explicit HMI confidence indices in [0,6].')
    arrays = {}
    for name in parameters:
        native_name = 'apparent_B_los' if name == 'B_los' else name
        arrays[name] = _sample_image(record['field']['path'], x, y, limits=(0, np.inf),
                                     confidence=record['confidence']['path'], accepted=confidence,
                                     magnetic=(record['inclination']['path'], native_name, sign))
    mask = np.logical_and.reduce([np.isfinite(values) for values in arrays.values()])
    return EvaluationResult(arrays, {name: 'deg' if name == 'inclination' else 'G' for name in arrays},
                            header, mask, {'model_time': record['time'], 'reference': 'HMI90 pipeline, filling factor=1',
                                           'sampling': 'Scalar native magnetic quantities sampled at output pixel centers; not pixel-integrated flux.'})


def _comparison_figure(prediction, reference, mask, parameter, output, run_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from spinnme.plotting import magnetic_norm, plot_comparison
    a, b = prediction.arrays[parameter], reference.arrays[parameter]
    unit = reference.units[parameter]
    limits = np.percentile(np.r_[a[mask], b[mask]], [1, 99])
    if limits[0] == limits[1]:
        limits += np.array([-1, 1])
    if parameter == 'B_los':
        limits = [-max(abs(limits)), max(abs(limits))]
    norm = magnetic_norm(parameter, np.r_[a[mask], b[mask]]) if parameter == 'B' else Normalize(*limits)
    fig = plot_comparison(prediction, reference, [parameter], masks=mask,
        norms={parameter: norm}, cmaps={parameter: 'RdBu_r' if parameter == 'B_los' else 'viridis'},
        source_labels=('HMI90 pipeline', run_id), difference_floor=.1 if unit == 'deg' else .5)
    fig.suptitle(parameter+'; '+reference.metadata['model_time']+' UTC')
    save_figure(fig, output/(parameter+'_'+run_id+'_maps'), ('pdf', 'svg', 'png'), 300)
    plt.close(fig)


def _summary(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output/'figures'
    per_run = defaultdict(list)
    for row in rows:
        per_run[(row['run_id'], row['cadence_s'], row['parameter'], row['unit'])].append(row)
    run_rows = []
    for key, group in per_run.items():
        run_rows.append({**dict(zip(('run_id', 'cadence_s', 'parameter', 'unit'), key)),
                         'n_epochs': len(group), 'bias': float(np.mean([row['bias'] for row in group])),
                         'mae': float(np.mean([row['mae'] for row in group])),
                         'rmse': float(np.sqrt(np.mean([row['rmse']**2 for row in group])))})
    _csv(output/'run_metrics.csv', run_rows)
    summary = []
    for row in run_rows:
        for metric in ('bias', 'mae', 'rmse'):
            summary.append({**{key: row[key] for key in ('run_id', 'cadence_s', 'parameter', 'unit')},
                            'metric': metric, 'mean': row[metric]})
    _csv(output/'summary.csv', summary)
    for parameter in sorted({row['parameter'] for row in summary}):
        selected = [row for row in summary if row['parameter'] == parameter and row['metric'] == 'rmse']
        labels = [f"{row['cadence_s']:g} s" for row in selected]
        fig, ax = plt.subplots(figsize=(max(6, .7*len(labels)), 3.5), layout='constrained')
        ax.plot(range(len(selected)), [row['mean'] for row in selected], 'o')
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
        ax.set(ylabel=f"RMSE [{selected[0]['unit']}]", title=parameter+': common HMI90 pipeline epochs')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        save_figure(fig, figures/(parameter+'_cadence'), ('pdf', 'svg', 'png'), 300)
        plt.close(fig)
        by_group = defaultdict(list)
        for row in rows:
            if row['parameter'] == parameter:
                by_group[(row['run_id'], row['cadence_s'])].append(row)
        origin = min(Time(row['time']).unix for row in rows)
        fig, ax = plt.subplots(figsize=(7.5, 4), layout='constrained')
        for (run_id, cadence), group in by_group.items():
            group.sort(key=lambda row: Time(row['time']).unix)
            ax.plot([(Time(row['time']).unix-origin)/60 for row in group], [row['rmse'] for row in group],
                    lw=1, label=f'{cadence:g} s')
        ax.set(xlabel='Time from first common epoch [min]', ylabel=f"RMSE [{selected[0]['unit']}]",
               title=parameter+': common HMI90 pipeline epochs')
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        save_figure(fig, figures/(parameter+'_temporal'), ('pdf', 'svg', 'png'), 300)
        plt.close(fig)
    return summary


def evaluate(runs_directory, reference_directory, output, *, device=DEVICE, batch_size=BATCH_SIZE, time=None,
             selected_runs=None):
    runs_directory, reference_directory, output = (Path(runs_directory).resolve(),
                                                 Path(reference_directory).resolve(), Path(output).resolve())
    runs = _runs(runs_directory, selected_runs)
    records = _references(reference_directory)
    start, stop = _training_span(runs)
    records = [row for row in records if start-1e-3 <= Time(row['time']).unix <= stop+1e-3]
    if not records:
        raise ValueError('No pipeline epochs lie in the common training span.')
    requested = time
    if requested:
        nearest = min(records, key=lambda row: abs(Time(row['time']).unix-Time(requested).unix))
        if abs(Time(nearest['time']).unix-Time(requested).unix) > 45:
            raise ValueError('No actual HMI90 pipeline epoch lies within 45 seconds of the requested time.')
        records = [nearest]
    else:
        stride = TIME_STRIDE
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError('time_stride must be a positive integer.')
        records = records[::stride]
    output.mkdir(parents=True, exist_ok=True)
    products = output/'products'
    products.mkdir(exist_ok=True)
    references = []
    for index, record in enumerate(records):
        reference = _reference(record)
        reference.write_npz(products/f'reference_{index:04d}.npz', overwrite=True)
        references.append(reference)
    parameters = list(references[0].arrays)
    for run in runs:
        with load(run['snapshot'], device=device) as snapshot:
            if 'HMI' in snapshot.module.limb_correction_models:
                raise ValueError('HMI pipeline comparison requires fixed filling factor one; use the baseline with limb_correction: false.')
            for index, reference in enumerate(references):
                observation = {'id': 'common-hmi-grid', 'instrument_id': 'HMI', 'shape': list(reference.mask.shape),
                               'header': dict(reference.header), 'time': {'model_time': reference.metadata['model_time']}}
                result = snapshot.evaluate(observation, instrument_id='HMI', parameters=parameters,
                                           time=reference.metadata['model_time'],
                                           batch_size=batch_size)
                result.write_npz(products/run['run_id']/f'frame_{index:04d}.npz', overwrite=True)
    rows = []
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    for index, reference in enumerate(references):
        predictions = [read_npz(products/run['run_id']/f'frame_{index:04d}.npz') for run in runs]
        common = reference.mask.copy()
        for result in predictions:
            common &= result.mask
            common &= np.logical_and.reduce([np.isfinite(result.arrays[name]) for name in parameters])
        if not common.any():
            raise ValueError('No common finite pixels across all inversions and the HMI pipeline.')
        mask_header = reference.header.copy()
        mask_header['CONTENT'] = 'common comparison mask across all runs'
        mask_header['BUNIT'] = '1'
        fits.writeto(products/f'common_mask_{index:04d}.fits', common.astype(np.uint8), mask_header,
                     overwrite=True, checksum=True)
        for run, result in zip(runs, predictions):
            for parameter in parameters:
                metric, _ = comparison(result.arrays[parameter], reference.arrays[parameter], common)
                rows.append({**{key: run[key] for key in ('run_id', 'cadence_s')},
                             'time': reference.metadata['model_time'], 'parameter': parameter,
                             'unit': reference.units[parameter], **metric})
                if index == len(references)//2:
                    _comparison_figure(result, reference, common, parameter, figures, run['run_id'])
    _csv(output/'frame_metrics.csv', rows)
    summary = _summary(rows, output)
    write_json(output/'comparison.json', {'epochs_utc': [row['time'] for row in records],
               'training_span_intersection_utc': Time([start, stop], format='unix').utc.isot.tolist(),
               'parameters': parameters, 'runs': [{**row, 'snapshot': str(row['snapshot'])} for row in runs],
               'comparison': 'HMI90 pipeline is an operational inversion reference, not ground truth.',
               'aggregation': 'Equal weight per common epoch within each native-cadence run.',
               'summary': summary})
    return output/'summary.csv'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path, help='Parent directory of the two HMI run folders')
    parser.add_argument('--run', action='append', choices=('hmi90', 'hmi720'), dest='selected_runs',
                        help='Run to compare; repeat to select multiple (default: both)')
    parser.add_argument('--reference', required=True, type=Path, help='Absolute HMI90 pipeline reference directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--time', help='Select the actual HMI90 reference epoch nearest this UTC time')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default=DEVICE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    try:
        print(evaluate(args.runs, args.reference, args.output, device=args.device, batch_size=args.batch_size,
                       time=args.time, selected_runs=args.selected_runs))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f'HMI cadence evaluation: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
