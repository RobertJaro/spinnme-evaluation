"""Compare native PHI-FDT inversions with acquisition-matched L2 pipeline maps."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from spinnme import load
from spinnme.data.fits import parse_time, spatial_header
from spinnme.data.phi import read_phi_observation
from spinnme.exports import read_npz
from spinnme.plotting import magnetic_norm
from _common import comparison, checksum, save_figure, utc_now, write_json

DATA = Path('/glade/work/rjarolim/data/phi_fdt/20240323_full_day')
RUN = Path('/glade/work/rjarolim/spinn_me/phi/fdt_20240323_v01')
OUTPUT = Path('/glade/work/rjarolim/spinn_me/evaluation/phi_fdt')
PRODUCTS = ('stokes', 'bmag', 'binc', 'bazi', 'blos', 'vlos')
UNITS = {'bmag': 'G', 'binc': 'deg', 'bazi': 'deg', 'blos': 'G', 'vlos': 'm/s'}
FORMATS = ('png', 'pdf')
DPI = 180
POLARIZATION_SNR = 5.


def header_and_shape(path):
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
        image = next((h for h in hdus if h.header.get('NAXIS') in (2, 4)), None)
        if image is None:
            raise ValueError(f'No PHI science image: {path}')
        header = hdus[0].header.copy()
        header.update(image.header)
    return header, (header['NAXIS2'], header['NAXIS1'])


def acquisition(header):
    if 'PHIDATID' not in header or 'DATE-OBS' not in header:
        raise ValueError('PHI pairing requires PHIDATID and DATE-OBS; no nearest-time matching is used.')
    return str(header['PHIDATID']).strip(), parse_time(header['DATE-OBS'], str(header.get('TIMESYS', 'UTC')).lower()).isoformat()


def same_grid(first, second, shape):
    # Comparing in pixel coordinates handles WCS longitude wrapping and units.
    for key, atol in (('DSUN_OBS', 1000.), ('HGLN_OBS', 1e-6), ('HGLT_OBS', 1e-6)):
        if key not in first or key not in second or not np.isclose(float(first[key]), float(second[key]), rtol=0, atol=atol):
            raise ValueError(f'PHI observer geometry differs or lacks {key}.')
    y, x = np.meshgrid(np.linspace(0, shape[0]-1, 3), np.linspace(0, shape[1]-1, 3), indexing='ij')
    a = WCS(spatial_header(first, shape)).celestial
    b = WCS(spatial_header(second, shape)).celestial
    xx, yy = b.world_to_pixel_values(*a.pixel_to_world_values(x, y))
    error = np.hypot(xx-x, yy-y)
    if not np.isfinite(error).all() or np.max(error) > 1e-3:
        raise ValueError('PHI spatial grids differ; register the products explicitly before comparison.')


def paired_products(data, version=3):
    """Require a single selected archive release and exact acquisition pairs."""
    indexed, headers = {}, {}
    for product in PRODUCTS:
        indexed[product] = {}
        for path in sorted((Path(data)/product).rglob('*.fits*')):
            if not path.is_file() or not path.name.endswith(('.fits', '.fits.gz', '.fits.fz')):
                continue
            h, shape = header_and_shape(path)
            if f'phi-fdt-{product}' not in path.name.lower():
                raise ValueError(f'Cannot verify FDT product identity from filename: {path}')
            if str(h.get('PHIDTYPE', product)).lower() not in (product, f'phi-fdt-{product}'):
                raise ValueError(f'Conflicting PHIDTYPE: {path}')
            try:
                actual_version = int(h['VERSION'])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f'Missing/invalid archive VERSION: {path}') from exc
            if actual_version != version:
                raise ValueError(f'Expected archive version {version}, found {actual_version}: {path}. Use a clean selected collection.')
            key = acquisition(h)
            if key in indexed[product]:
                raise ValueError(f'Duplicate PHI {product} acquisition {key}; select one processing version.')
            if product == 'stokes' and (h.get('NAXIS') != 4 or h.get('NAXIS3') != 4 or h.get('NAXIS4') != 6):
                raise ValueError(f'Expected current-release FDT wavelength,Stokes,y,x cube: {path}')
            if product != 'stokes':
                if h.get('NAXIS') != 2:
                    raise ValueError(f'Expected a scalar pipeline map: {path}')
                if 'BUNIT' not in h:
                    raise ValueError(f'Pipeline BUNIT is required: {path}')
                u.Unit(h['BUNIT']).to(u.Unit(UNITS[product]))
            indexed[product][key] = path
            headers[path] = (h, shape)
        if not indexed[product]:
            raise ValueError(f'No {product} files found in {Path(data)/product}.')
    keys = set(indexed['stokes'])
    for product in PRODUCTS[1:]:
        if set(indexed[product]) != keys:
            raise ValueError(f'Unpaired {product}: missing={sorted(keys-set(indexed[product]))}, extra={sorted(set(indexed[product])-keys)}')
    records = []
    for key in sorted(keys, key=lambda k: k[1]):
        files = {p: indexed[p][key] for p in PRODUCTS}
        h, shape = headers[files['stokes']]
        # Require matching temporal integration, not merely a shared start time.
        for product in PRODUCTS[1:]:
            other, other_shape = headers[files[product]]
            if other_shape != shape:
                raise ValueError(f'PHI shapes differ for {key}: {product}')
            same_grid(h, other, shape)
            for card in ('DATE-END', 'DATE-AVG', 'ACCRAWIT'):
                if (card in h) != (card in other) or (card in h and h[card] != other[card]):
                    raise ValueError(f'PHI integration metadata {card} differs for {key}: {product}')
        records.append({'id': key[0], 'time': key[1], 'version': version, 'files': files,
                        'shape': list(shape), 'onboard_averages': h.get('ACCRAWIT'),
                        'quality': {p: headers[path][0].get('QUALITY') for p, path in files.items()}})
    return records


def pipeline_maps(record):
    maps = {}
    for product in PRODUCTS[1:]:
        path = record['files'][product]
        with fits.open(path, memmap=False) as hdus:
            image = next(h for h in hdus if h.header.get('NAXIS') == 2)
            h = hdus[0].header.copy()
            h.update(image.header)
            maps[product] = (np.asarray(image.data, dtype=float)*u.Unit(h['BUNIT'])).to_value(UNITS[product])
    # Nonlinear conversion on the native grid; no interpolation of B/gamma.
    return {'B': maps['bmag'], 'B_los': maps['blos'],
            'B_transverse': maps['bmag']*np.sin(np.deg2rad(maps['binc'])),
            'inclination': maps['binc'], 'azimuth': maps['bazi'] % 180,
            'forward_vdop': maps['vlos']}


def paired_mask(result, reference, observed, continuum_index):
    valid = result.mask & np.isfinite(observed).all(axis=(-2, -1))
    valid &= observed[..., 0, continuum_index] > 0
    for key, values in reference.items():
        valid &= np.isfinite(values) & np.isfinite(result.arrays[key])
    valid &= (reference['B'] >= 0) & (reference['inclination'] >= 0) & (reference['inclination'] <= 180)
    return valid


def diagnostics(result, reference, observed, continuum_index):
    """Common-mask metrics, robust noise estimates and explicit velocity offset."""
    valid = paired_mask(result, reference, observed, continuum_index)
    if not valid.any():
        raise ValueError('No common finite on-disk pixels remain.')
    noise = []
    for index in (1, 2, 3):
        values = observed[..., index, continuum_index][valid]
        noise.append(float(1.4826*np.median(np.abs(values-np.median(values)))))
    linear = np.max(np.hypot(observed[..., 1, :], observed[..., 2, :]), axis=-1)
    total = np.max(np.linalg.norm(observed[..., 1:, :], axis=-2), axis=-1)
    angle_masks = {'azimuth': linear > POLARIZATION_SNR*max(np.hypot(*noise[:2]), np.finfo(float).eps),
                   'inclination': total > POLARIZATION_SNR*max(np.linalg.norm(noise), np.finfo(float).eps)}
    rows, comparisons = [], {}
    for key, target in reference.items():
        predicted = result.arrays[key]
        selected = valid & angle_masks.get(key, True)
        removed = None
        if key == 'forward_vdop':
            # Do not compare observer-corrected solar Vdop directly with the
            # operational pipeline vlos. Only a relative velocity comparison
            # is made; the common additive offset is separately preserved.
            removed = float(np.median((predicted-target)[valid]))
            predicted = predicted-removed
        period = 180 if key == 'azimuth' else None
        if selected.any():
            metric, difference = comparison(predicted, target, selected, period=period)
        else:
            metric, difference = {'n': 0, 'status': 'no_pixels_above_polarization_threshold'}, np.full(valid.shape, np.nan)
        name = 'vlos_relative' if key == 'forward_vdop' else key
        unit = 'm/s' if key == 'forward_vdop' else 'deg' if key in angle_masks else 'G'
        rows.append({'parameter': name, 'unit': unit, 'removed_velocity_offset_m_s': removed, **metric})
        comparisons[name] = (np.where(selected, predicted, np.nan), np.where(selected, target, np.nan), difference, unit)
    stokes_rows = []
    for index, key in enumerate('IQUV'):
        target, predicted = observed[..., index, :], result.arrays[key]
        if predicted.shape != target.shape or not np.isfinite(predicted[valid]).all():
            raise ValueError(f'Exported {key} spectra have the wrong shape or non-finite predictions on valid pixels.')
        selected = valid[..., None] & np.isfinite(predicted)
        residual = np.where(selected, predicted-target, np.nan)
        stokes_rows.append({'stokes': key, 'n': int(np.count_nonzero(selected)),
                           'bias': float(np.nanmean(residual)), 'rmse': float(np.sqrt(np.nanmean(residual**2)))})
    return rows, stokes_rows, comparisons, valid, noise


def figures(result, observed, comparisons, valid, offsets, output):
    import matplotlib.pyplot as plt
    for key, (predicted, target, difference, unit) in comparisons.items():
        if not np.isfinite(difference).any():
            continue
        figure, axes = plt.subplots(1, 4, figsize=(17, 4), layout='constrained')
        low, high = np.nanpercentile(np.stack([predicted, target]), [1, 99])
        if key in ('B_los', 'vlos_relative'):
            high = max(abs(low), abs(high), 1e-6)
            low = -high
        bound = max(float(np.nanpercentile(np.abs(difference), 99)), 1e-6)
        for axis, values, title, limits in zip(axes[:3], (target, predicted, difference),
                ('PHI pipeline', 'SPINNME', 'SPINNME − pipeline'), ((low, high), (low, high), (-bound, bound))):
            from matplotlib.colors import Normalize
            norm = magnetic_norm(key, np.r_[predicted.ravel(), target.ravel()]) if not title.startswith('SPINNME −') else None
            image = axis.imshow(values, origin='lower', norm=norm if norm is not None else Normalize(*limits),
                                cmap='RdBu_r' if title.startswith('SPINNME −') or low < 0 else 'viridis')
            axis.set_title(title)
            axis.set_xlabel('Native detector x [pixel]')
            axis.set_ylabel('Native detector y [pixel]')
            figure.colorbar(image, ax=axis, label=unit, shrink=.8)
        mask = np.isfinite(difference)
        axes[3].hexbin(target[mask], predicted[mask], gridsize=80, bins='log', mincnt=1)
        axes[3].plot([low, high], [low, high], 'k--', linewidth=.8)
        axes[3].set_xlabel(f'PHI pipeline [{unit}]')
        axes[3].set_ylabel(f'SPINNME [{unit}]')
        figure.suptitle(f'{key} · {result.metadata["model_time"]} · operational reference, not truth')
        save_figure(figure, output/key, FORMATS, DPI)
        plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
    order = np.argsort(offsets)
    for index, key in enumerate('IQUV'):
        target, predicted = observed[..., index, :][valid], result.arrays[key][valid]
        axes[0, index].plot(offsets[order], np.mean(target, axis=0)[order], 'o-', label='observed')
        axes[0, index].plot(offsets[order], np.mean(predicted, axis=0)[order], 'o-', label='SPINNME')
        axes[0, index].set_title(key)
        axes[0, index].set_ylabel('Continuum-normalized intensity')
        axes[0, index].legend()
        axes[1, index].plot(offsets[order], np.sqrt(np.mean((predicted-target)**2, axis=0))[order], 'o-')
        axes[1, index].set_ylabel('RMS fit residual')
        axes[1, index].set_xlabel('Calibrated line offset [Å]')
    save_figure(figure, output/'stokes', FORMATS, DPI)
    plt.close(figure)


def compare(data=DATA, snapshot=RUN/'inversion.pme', products=RUN/'products', output=OUTPUT, *, version=3):
    records = paired_products(data, version)
    by_source = {str(row['files']['stokes'].resolve()): row for row in records}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new or empty comparison output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    rows, stokes_rows, provenance, seen = [], [], [], set()
    with load(snapshot, device='cpu') as model:
        observations = {row['id']: row for row in model.observations if row.get('role') == 'train'}
        for path in sorted(Path(products).glob('*/parameters.npz')):
            result = read_npz(path)
            identifier = result.metadata['observation_id']
            if identifier not in observations:
                continue
            obs = observations[identifier]
            if result.metadata.get('config_hash') != model.metadata['config_hash']:
                raise ValueError('Export was produced with a different inversion configuration.')
            if obs.get('dataset_type') != 'phi-fdt' or obs.get('time', {}).get('time_reference') != 'observation':
                raise ValueError('Comparison requires native FDT observations with time_reference: observation.')
            instrument = obs['instrument_id']
            if instrument in model.limb_correction_models or instrument in model.artifact_correction_models or instrument in model.velocity_correction_models:
                raise ValueError('Use the baseline FDT inversion without learned instrument/filling-factor corrections.')
            if model.module.spectral_responses[instrument].kind == 'monochromatic':
                raise ValueError('FDT comparison requires the configured finite spectral response.')
            if not model.continuum_normalization_mode:
                raise ValueError('This workflow requires the continuum-normalized FDT preset.')
            source = str(Path(obs['source_paths'][0]).resolve())
            if source not in by_source or source in seen:
                raise ValueError(f'Unmatched or duplicate exported Stokes acquisition: {source}')
            seen.add(source)
            record = by_source[source]
            native_header, shape = header_and_shape(source)
            saved_header = fits.Header.fromstring(obs['original_header'], sep='\n')
            if acquisition(saved_header) != acquisition(native_header) or saved_header.get('VERSION') != native_header.get('VERSION'):
                raise ValueError('Source acquisition/release no longer matches the snapshot.')
            if tuple(result.mask.shape) != shape:
                raise ValueError('Only native-grid FDT exports can be compared.')
            same_grid(native_header, result.header, shape)
            if abs((Time(result.metadata['model_time'])-Time(record['time'])).to_value(u.s)) > .001:
                raise ValueError('Export and pipeline acquisition epochs differ.')
            reference = pipeline_maps(record)
            for key in reference:
                if u.Unit(result.units[key]) != u.Unit('m/s' if key == 'forward_vdop' else 'deg' if key in ('azimuth', 'inclination') else 'G'):
                    raise ValueError(f'Unexpected exported {key} units.')
            data_obs = read_phi_observation(source, telescope='fdt', **obs.get('reader', {}))
            observed = data_obs['stokes']/float(obs['stokes_normalization'])
            observed = model.module.stokes_loss.normalize_profile(torch.from_numpy(observed)).numpy()
            offsets = data_obs['wavelength_config']['wavelength_grid'].to_value(u.m)
            if not np.allclose(offsets, result.metadata['wavelength_grid_m'], rtol=0, atol=1e-18):
                raise ValueError('Observed and exported wavelength grids differ.')
            if not np.isclose(data_obs['wavelength_config']['wavelength_center'].to_value(u.m),
                              result.metadata['wavelength_center_m'], rtol=0, atol=1e-18):
                raise ValueError('Observed and exported reference wavelengths differ.')
            metrics, stokes, maps, mask, noise = diagnostics(result, reference, observed, data_obs['continuum_index'])
            rows.extend({'observation': identifier, 'time': record['time'], **row} for row in metrics)
            stokes_rows.extend({'observation': identifier, **row} for row in stokes)
            own = output/identifier
            own.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(own/'masks.npz', common=mask, **{key: np.isfinite(values[2]) for key, values in maps.items()})
            figures(result, observed, maps, mask, offsets*1e10, own)
            provenance.append({**record, 'noise_QUV': noise, 'export': str(path), 'export_sha256': checksum(path),
                               'sha256': {key: checksum(file) for key, file in record['files'].items()}})
    if seen != set(by_source):
        raise ValueError('Exports are missing one or more downloaded Stokes acquisitions; export all trained observations.')
    for name, values in (('metrics', rows), ('stokes_metrics', stokes_rows)):
        fields = list(dict.fromkeys(key for row in values for key in row))
        with (output/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    write_json(output/'comparison.json', {'created': utc_now(), 'snapshot': str(snapshot), 'snapshot_sha256': checksum(snapshot),
        'records': provenance, 'metrics': rows, 'stokes_metrics': stokes_rows, 'polarization_snr': POLARIZATION_SNR,
        'reference': 'Operational PHI-FDT inversion, not ground truth. No azimuth disambiguation.',
        'velocity': 'forward_vdop minus pipeline vlos, with one common-mask median offset removed per acquisition; offset retained in metrics. No absolute solar velocity claim.',
        'calibration': 'Nominal finite filtergraph response; no spatial PSF matching or residual cavity correction.'})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--snapshot', type=Path, default=RUN/'inversion.pme')
    parser.add_argument('--products', type=Path, default=RUN/'products')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--version', type=int, default=3, help='Required archive VERSION, not the SPINNME run suffix')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.check_only:
        records = paired_products(args.data, args.version)
        write_json(args.data/'pairs.json', {'records': records})
        print(f'Verified {len(records)} FDT acquisition(s); pairing details: {args.data / "pairs.json"}')
    else:
        print(compare(args.data, args.snapshot, args.products, args.output, version=args.version))


if __name__ == '__main__':
    main()
