"""Publication plates: HMI 90/720 s and SPINNME 90/720 s, full disk and HPC zoom.

The 720 s reference supplies the common observing grid and model epoch. The
nearest real 90 s reference is retained at its actual epoch (no interpolation).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from astropy import units as u
from astropy.time import Time
from sunpy.map import all_coordinates_from_map, coordinate_is_on_solar_disk

from _common import _header, _model_time, _record_key, _sample_image, _series, _world
from hmi_reference import HMI_UNITS, reference_grid

PARAMETERS = ('B', 'inclination', 'azimuth')
TITLES = ('HMI reference 90 s', 'HMI reference 720 s', 'SPINNME 90 s', 'SPINNME 720 s')
ZOOM_HPC = (0., 100., 0., 50.)  # XMIN, XMAX, YMIN, YMAX in arcsec.


def reference_record(directory, cadence, time=None):
    """Require exactly three paired magnetic segments from the requested series."""
    directory = Path(directory)
    if directory.is_file():
        if not directory.name.endswith('.field.fits'):
            raise ValueError('A reference file path must select the field FITS segment (*.field.fits).')
        fields = [(directory, _header(directory)[0])]
        directory = directory.parent
    else:
        fields = [(path, _header(path)[0]) for path in sorted(directory.glob('*.field.fits'))]
    if not fields:
        raise FileNotFoundError(f'No *.field.fits in {directory}')
    fields.sort(key=lambda row: _model_time(row[1]).unix)
    path, header = (fields[len(fields)//2] if time is None else
                    min(fields, key=lambda row: abs((_model_time(row[1])-Time(time)).to_value(u.s))))
    epoch = _model_time(header)
    if time is not None and abs((epoch-Time(time)).to_value(u.s)) > cadence/2+1:
        raise ValueError(f'No HMI {cadence} s epoch within half a cadence of {time}.')
    key = _record_key(header)
    files = {'field': path}
    for segment in ('inclination', 'azimuth'):
        matches = [item for item in sorted(directory.glob(f'*.{segment}.fits'))
                   if _record_key(_header(item)[0]) == key]
        if len(matches) != 1:
            raise ValueError(f'Expected one {segment} for HMI {cadence} s record {key}; found {len(matches)}.')
        files[segment] = matches[0]
    for item in files.values():
        own, _ = _header(item)
        _series(own, item, f'hmi.B_{cadence}s')
        if int(own.get('QUALITY', 0)) != 0:
            raise ValueError(f'Nonzero HMI QUALITY: {item}')
    return files, epoch


def sample_reference(files, grid):
    """Nearest native samples preserve angular branches and missing pixels."""
    x, y = _world(grid.fits_header, grid.data.shape)
    disk = coordinate_is_on_solar_disk(all_coordinates_from_map(grid))
    arrays = {}
    for segment, name, unit in (('field', 'B', u.G), ('inclination', 'inclination', u.deg),
                                ('azimuth', 'azimuth', u.deg)):
        header, _ = _header(files[segment])
        declared = str(header.get('BUNIT') or '').strip()
        if declared.lower() in ('', 'none', 'undefined', 'unknown', 'n/a', '?'):
            declared = HMI_UNITS[segment]
        declared = {'gauss': 'G', 'degrees': 'deg', 'degree': 'deg', 'radians': 'rad'}.get(declared.lower(), declared)
        values = _sample_image(files[segment], x, y, nearest=True)*u.Unit(declared).to(unit)
        valid = disk & np.isfinite(values)
        if name == 'B':
            valid &= values >= 0
        elif name == 'inclination':
            valid &= (values >= 0) & (values <= 180)
        else:
            values %= 180
        arrays[name] = np.where(valid, values, np.nan)
    return arrays


def color_scales(bmax):
    return ((LogNorm(1, bmax, clip=True), 'viridis', r'$|B|$ [G]'),
            (Normalize(0, 180), 'PiYG', r'Inclination [$^\circ$]'),
            (Normalize(0, 180), cmr.infinity, r'Azimuth mod $180^\circ$ [$^\circ$]'))


def publication_figure(columns, header, bounds, scales, *, titles=False, zoom_box=None, frame=False):
    """Three rows, four columns, one colorbar per row; exact HPC pixel edges."""
    shape = columns[0]['B'].shape
    edge_header = header.copy()
    edge_header['CRPIX1'] += .5
    edge_header['CRPIX2'] += .5
    x, y = _world(edge_header, (shape[0]+1, shape[1]+1))
    xmin, xmax, ymin, ymax = bounds
    panel_height = 1.65*(ymax-ymin)/(xmax-xmin)
    # Let labels and titles determine spacing; reserve the last column for bars.
    fig = plt.figure(figsize=(9.4, 3*max(panel_height, 1.55)+1.0), layout='constrained')
    gs = fig.add_gridspec(3, 5, width_ratios=(1, 1, 1, 1, .045))
    for row, (name, (norm, cmap, label)) in enumerate(zip(PARAMETERS, scales)):
        for col, arrays in enumerate(columns):
            ax = fig.add_subplot(gs[row, col])
            image = ax.pcolormesh(x, y, np.ma.masked_invalid(arrays[name]), norm=norm, cmap=cmap,
                                  shading='flat', rasterized=True)
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')
            ax.spines[['top', 'right']].set_visible(frame)
            ax.tick_params(top=False, right=False, labelbottom=row == 2, labelleft=col == 0,
                           direction='out', length=3, pad=2, labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune='both'))
            if row == 2:
                ax.set_xlabel('Solar X [arcsec]', fontsize=9)
            if col == 0:
                ax.set_ylabel('Solar Y [arcsec]', fontsize=9)
            if titles and row == 0:
                ax.set_title(TITLES[col], fontsize=10, pad=9)
            if zoom_box is not None:
                x0, x1, y0, y1 = zoom_box
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                       edgecolor='red', linewidth=.9, zorder=5))
        cax = fig.add_subplot(gs[row, 4])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(label, fontsize=9)
        bar.ax.tick_params(labelsize=8)
        if row:
            bar.set_ticks([0, 45, 90, 135, 180])
    return fig


def evaluate(input90, input720, reference90, reference720, output, *, time=None,
             zoom=ZOOM_HPC, resolution=1024, zoom_resolution=0, bmax=3000., dpi=400,
             device='auto', batch_size=8192):
    from spinnme import load
    from spinnme.exports import EvaluationResult

    zoom = np.asarray(zoom, dtype=float)
    if zoom.shape != (4,) or not np.isfinite(zoom).all() or zoom[0] >= zoom[1] or zoom[2] >= zoom[3]:
        raise ValueError('zoom must be finite XMIN XMAX YMIN YMAX with positive extent.')
    if not np.isfinite(bmax) or bmax <= 1:
        raise ValueError('bmax must be greater than 1 G.')
    if dpi < 1 or batch_size < 1 or any(value < 0 or value == 1 for value in (resolution, zoom_resolution)):
        raise ValueError('dpi/batch-size must be positive; resolutions must be zero or >=2.')
    scales = color_scales(bmax)
    files720, epoch = reference_record(reference720, 720, time)
    files90, epoch90 = reference_record(reference90, 90, epoch)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with load(input90, device=device) as model90, load(input720, device=device) as model720:
        for name, region, size in (('full_disk', None, resolution), ('zoom', zoom, zoom_resolution)):
            grid, _, observation = reference_grid(files720, model90, region, size)
            columns = [sample_reference(files, grid) for files in (files90, files720)]
            for model in (model90, model720):
                result = model.evaluate(observation, instrument_id='HMI', time=epoch.utc.isot,
                                        parameters=PARAMETERS, batch_size=batch_size)
                columns.append({key: np.where(result.mask, result.arrays[key] % 180 if key == 'azimuth'
                                               else result.arrays[key], np.nan) for key in PARAMETERS})
            for col, arrays in enumerate(columns):
                mask = np.logical_and.reduce([np.isfinite(arrays[key]) for key in PARAMETERS])
                if not mask.any():
                    raise ValueError(f'No finite on-disk data in {name}, {TITLES[col]}.')
                EvaluationResult(arrays, {'B': 'G', 'inclination': 'deg', 'azimuth': 'deg'},
                                 grid.fits_header, mask, {'source': TITLES[col],
                                 'model_time': (epoch90 if col == 0 else epoch).utc.isot}).write_npz(
                                     output/f'{name}_{col+1}.npz', overwrite=True)
            radius = grid.rsun_obs.to_value(u.arcsec)*1.08
            bounds = (-radius, radius, -radius, radius) if region is None else zoom.tolist()
            fig = publication_figure(columns, grid.fits_header, bounds, scales,
                                     titles=name == 'full_disk', zoom_box=zoom if name == 'full_disk' else None,
                                     frame=name == 'zoom')
            # Constrained layout reserves space for all axis and colorbar labels.
            fig.savefig(output/f'cadence_{name}.png', dpi=dpi)
            plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for cadence in (90, 720):
        parser.add_argument(f'--input{cadence}', type=Path, required=True,
                            help=f'SPINNME {cadence} s .pme snapshot path')
        parser.add_argument(f'--reference{cadence}', type=Path, required=True,
                            help=f'HMI {cadence} s field FITS path or directory of paired reference FITS')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time', help='UTC epoch nearest the HMI720 record (default: middle record)')
    parser.add_argument('--zoom', type=float, nargs=4, default=ZOOM_HPC,
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'), help='HPC arcsec; default: 0 100 0 50')
    parser.add_argument('--resolution', type=int, default=1024, help='Full-disk maximum dimension; 0=native')
    parser.add_argument('--zoom-resolution', type=int, default=0, help='Zoom maximum dimension; 0=native (default)')
    parser.add_argument('--bmax', type=float, default=3000., help='Shared field upper limit in G (default: 3000)')
    parser.add_argument('--dpi', type=int, default=400)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        print(evaluate(**vars(args)))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f'HMI cadence comparison: {exc}\n')


if __name__ == '__main__':
    main()
