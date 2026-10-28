"""Plot native PHI-HRT exports and Stokes residuals; no training or downloading."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.wcs import WCS

from _common import save_figure
from spinnme import load
from spinnme.data.phi import read_phi_observation
from spinnme.exports import read_npz
from spinnme.plotting import magnetic_norm


SNAPSHOT = '/glade/work/rjarolim/spinn_me/phi/hrt_20240323_v01/inversion.pme'
PARAMETERS = ('Br', 'Btheta', 'Bphi', 'Blos', 'Vdop')
ARTIFACT_PARAMETERS = ('artifact_alpha_Q', 'artifact_alpha_U', 'artifact_alpha_V', 'artifact_gate')
FORMATS = ('pdf', 'svg', 'png')
DPI = 300


def plot(products, output, snapshot=SNAPSHOT, *, device=None):
    import matplotlib.pyplot as plt
    products, output = Path(products), Path(output)
    files = sorted(products.glob('*/parameters.npz'))
    if not files:
        raise FileNotFoundError(f'No exported PHI NPZ maps under {products}; run scripts/phi/evaluation.sh to export first.')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with load(snapshot, device=device) as model:
        observations = {row['id']: row for row in model.observations}
        for path in files:
            result = read_npz(path)
            identifier = result.metadata['observation_id']
            observation = observations[identifier]
            # Training and validation may point to the same source; plot it once.
            if observation.get('role') == 'validation':
                continue
            own = output/identifier
            own.mkdir(parents=True, exist_ok=True)
            for parameters, name in ((PARAMETERS, 'maps'), (ARTIFACT_PARAMETERS, 'artifact')):
                figure, axes = plt.subplots(1, len(parameters), figsize=(4*len(parameters), 4),
                                            subplot_kw={'projection': WCS(result.header).celestial},
                                            layout='constrained', squeeze=False)
                for axis, key in zip(axes[0], parameters):
                    values = np.where(result.mask, result.arrays[key], np.nan)
                    finite = values[np.isfinite(values)]
                    if not len(finite):
                        raise ValueError(f'No valid {key} pixels in {path}.')
                    low, high = np.percentile(finite, [1, 99])
                    signed = key != 'artifact_gate'
                    if signed:
                        high = max(abs(low), abs(high), 1e-12)
                        low = -high
                    from matplotlib.colors import Normalize
                    norm = magnetic_norm(key, finite)
                    image = axis.imshow(values, origin='lower', norm=norm if norm is not None else Normalize(low, high),
                                        cmap='RdBu_r' if signed else 'viridis')
                    axis.set_title(key)
                    figure.colorbar(image, ax=axis, label=result.units[key], shrink=.8)
                save_figure(figure, own/name, FORMATS, DPI)
                plt.close(figure)
            reader = observation.get('reader', {})
            data = read_phi_observation(observation['source_paths'][0], telescope='hrt', **reader)
            observed = data['stokes']/float(observation['stokes_normalization'])
            if model.continuum_normalization_mode == 'per_profile':
                observed = model.module.stokes_loss.normalize_profile(torch.from_numpy(observed)).numpy()
            if observed.shape[:2] != result.mask.shape:
                raise ValueError('PHI residuals require native-grid exports matching the source Stokes cube.')
            offsets = data['wavelength_config']['wavelength_grid'].to_value(u.AA)
            if not np.allclose(offsets*1e-10, result.metadata['wavelength_grid_m'], rtol=1e-9, atol=1e-18):
                raise ValueError('PHI source and prediction wavelength grids differ.')
            figure, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
            for index, key in enumerate('IQUV'):
                prediction = result.arrays[key]
                target = observed[..., index, :]
                if prediction.shape != target.shape:
                    raise ValueError(f'{key} prediction/source shapes differ.')
                valid = result.mask[..., None] & np.isfinite(prediction) & np.isfinite(target)
                if not valid.any():
                    raise ValueError(f'No common finite {key} samples.')
                residual = np.where(valid, prediction-target, np.nan)
                axes[0, index].plot(offsets, np.nanmean(np.where(valid, target, np.nan), axis=(0, 1)), label='observed')
                axes[0, index].plot(offsets, np.nanmean(np.where(valid, prediction, np.nan), axis=(0, 1)), label='SPINNME')
                axes[0, index].set_title(key)
                axes[0, index].set_xlabel('Line offset [Å]')
                axes[0, index].set_ylabel('Continuum-normalized intensity')
                axes[0, index].legend()
                axes[1, index].hist(residual[valid], bins=80, histtype='step', density=True)
                axes[1, index].set_xlabel('SPINNME − observed')
                axes[1, index].set_ylabel('Density')
                rows.append({'observation': identifier, 'stokes': key, 'count': int(valid.sum()),
                             'bias': float(np.nanmean(residual)),
                             'rmse': float(np.sqrt(np.nanmean(residual**2)))})
            save_figure(figure, own/'stokes', FORMATS, DPI)
            plt.close(figure)
    if not rows:
        raise ValueError('No training observations were selected for PHI plots.')
    with (output/'stokes_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--products', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--snapshot', default=SNAPSHOT, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default=None)
    args = parser.parse_args(argv)
    print(plot(args.products, args.output, args.snapshot, device=args.device))


if __name__ == '__main__':
    main()
