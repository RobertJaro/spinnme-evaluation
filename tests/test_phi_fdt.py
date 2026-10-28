"""Offline scientific pairing, conventions, and full FDT workflow smoke test."""
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.map import make_fitswcs_header

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
workflow = importlib.import_module("phi_fdt")


@pytest.fixture
def collection(tmp_path):
    shape = (8, 8)
    observer = SkyCoord(lon=0*u.deg, lat=0*u.deg, radius=.4*u.AU,
                        obstime='2024-03-23T22:15:00', frame='heliographic_stonyhurst')
    center = SkyCoord(0*u.arcsec, 0*u.arcsec, observer=observer, obstime=observer.obstime,
                      frame='helioprojective')
    header = fits.Header(make_fitswcs_header(shape, center, scale=[3, 3]*u.arcsec/u.pix))
    header.update(PHIDATID='test001', VERSION='03', CONTPOS=6, WAVELNTH=6173.341,
                  WAVEUNIT='Angstrom', INSTRUME='PHI', TELESCOP='SO/PHI-FDT',
                  OBS_VR=0., OBS_VW=0., OBS_VN=0.)
    offsets = np.array([-.14, -.07, 0, .07, .14, .3])
    for i, offset in enumerate(offsets):
        header[f'WAVELN{i+1:02d}'] = header['WAVELNTH']+offset
    y, x = np.indices(shape)
    data = np.zeros((*shape, 4, 6), dtype=np.float32)
    data[..., 0, :] = 1-.3*np.exp(-(offsets/.08)**2)
    for index in (1, 2, 3):
        data[..., index, :] = .01*index*np.exp(-(offsets/.1)**2)
        data[..., index, -1] = ((x+y) % 3-1)*1e-4
    scalar = {'bmag': (500+10*x, 'G'), 'binc': (np.full(shape, 60.), 'deg'),
              'bazi': (np.full(shape, 179.), 'deg'), 'blos': (250+5*x, 'G'),
              'vlos': (x*.01+.5, 'km/s')}
    for product in workflow.PRODUCTS:
        own = tmp_path/'data'/product
        own.mkdir(parents=True)
        h = header.copy()
        h['PHIDTYPE'] = product
        values = data.transpose(3, 2, 0, 1) if product == 'stokes' else scalar[product][0]
        if product != 'stokes':
            h['BUNIT'] = scalar[product][1]
        fits.writeto(own/f'solo_L2_phi-fdt-{product}_test.fits', values, h)
    return tmp_path/'data'


def test_pairing_units_and_nonlinear_transverse_field(collection):
    record, = workflow.paired_products(collection)
    maps = workflow.pipeline_maps(record)
    np.testing.assert_allclose(maps['forward_vdop'][0, :2], [500, 510])
    np.testing.assert_allclose(maps['B_transverse'], maps['B']*np.sin(np.pi/3))


@pytest.mark.parametrize('change, message', [('missing', 'Unpaired'), ('duplicate', 'Duplicate'),
    ('version', 'Expected archive version'), ('grid', 'spatial grids differ'),
    ('epoch', 'Unpaired'), ('exposure', 'integration metadata')])
def test_incompatible_collections_are_rejected(collection, change, message):
    path = next((collection/'blos').glob('*.fits'))
    if change == 'missing':
        path.unlink()
        # Leave one differently timed reference so this tests pairing rather than an empty folder.
        source = next((collection/'bmag').glob('*.fits'))
        shutil.copyfile(source, path)
        with fits.open(path, mode='update') as hs:
            hs[0].header['PHIDTYPE'] = 'blos'
            hs[0].header['DATE-OBS'] = '2024-03-24T00:00:00'
    elif change == 'duplicate':
        shutil.copyfile(path, path.with_name('duplicate_phi-fdt-blos_test.fits'))
    else:
        with fits.open(path, mode='update') as hs:
            h = hs[0].header
            if change == 'version':
                h['VERSION'] = '02'
            elif change == 'grid':
                h['CRPIX1'] += 1
            elif change == 'epoch':
                h['DATE-OBS'] = '2024-03-24T00:00:00'
            else:
                h['ACCRAWIT'] = 3
    with pytest.raises(ValueError, match=message):
        workflow.paired_products(collection)


def test_wrapped_azimuth_and_velocity_zero_point(collection):
    from spinnme.data.phi import read_phi_observation
    from spinnme.exports import EvaluationResult
    record, = workflow.paired_products(collection)
    target = workflow.pipeline_maps(record)
    predicted = {key: values.copy() for key, values in target.items()}
    predicted['azimuth'][:] = 1
    predicted['forward_vdop'] += 1234
    observed = read_phi_observation(record['files']['stokes'], telescope='fdt')['stokes']
    predicted.update({key: observed[..., i, :] for i, key in enumerate('IQUV')})
    result = EvaluationResult(predicted, {}, fits.Header(), np.ones((8, 8), bool), {})
    rows, stokes, _, _, _ = workflow.diagnostics(result, target, observed, 5)
    rows = {row['parameter']: row for row in rows}
    assert rows['azimuth']['mae'] == pytest.approx(2)
    assert rows['vlos_relative']['rmse'] == pytest.approx(0)
    assert rows['vlos_relative']['removed_velocity_offset_m_s'] == pytest.approx(1234)
    assert all(row['rmse'] == 0 for row in stokes)


def test_training_export_comparison_smoke(collection, tmp_path, monkeypatch):
    import yaml
    from spinnme import load
    from spinnme.training.runner import run
    config = yaml.safe_load((ROOT/'configs/phi/fdt.yaml').read_text())
    config['path'], config['work_path'] = str(tmp_path/'run'), str(tmp_path/'cache')
    for dataset in (config['data']['train_configs'][0], config['data']['valid_config']):
        dataset['data_path'] = str(collection/'stokes')
    config['data'].update(batch_size=16, dataset_batch_size=8, num_dataset_workers=0)
    config['model'].update(dim=8, n_layers=1, encoding_config={'type': 'identity'})
    config['training'].update(max_steps=1, val_check_interval=1, plot_validation=False,
                              accelerator='cpu', devices=1, optimizer_config={'type': 'adamw'})
    config['logging']['mode'] = 'disabled'
    run(config)
    snapshot = tmp_path/'run/inversion.pme'
    products = tmp_path/'products'
    parameters = ['B', 'B_los', 'B_transverse', 'inclination', 'azimuth', 'forward_vdop', 'IQUV']
    with load(snapshot, device='cpu') as model:
        for row in model.observations:
            if row['role'] == 'train':
                result = model.evaluate(row['id'], parameters=parameters, batch_size=16, mu_limit=.2)
                result.write_npz(products/row['id'])
    monkeypatch.setattr(workflow, 'FORMATS', ('png',))
    output = workflow.compare(collection, snapshot, products, tmp_path/'comparison')
    report = json.loads((output/'comparison.json').read_text())
    assert {row['parameter'] for row in report['metrics']} == {
        'B', 'B_los', 'B_transverse', 'inclination', 'azimuth', 'vlos_relative'}
    assert len(report['stokes_metrics']) == 4
    assert list(output.glob('*/stokes.png'))
    assert (output/'metrics.csv').is_file()
