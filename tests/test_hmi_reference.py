"""Exercise native HMI conventions against a real portable SPINNME snapshot."""
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.map import Map, make_fitswcs_header

from spinnme import load
from spinnme.snapshots import save_snapshot
from spinnme.train.me_spherical_module import MESphericalModule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
workflow = importlib.import_module('hmi_reference')


@pytest.fixture
def collection(tmp_path):
    torch.manual_seed(3)
    shape = (13, 15)
    epoch = '2024-03-23T02:00:00'
    center = SkyCoord(0*u.arcsec, 0*u.arcsec, observer='earth', obstime=epoch, frame='helioprojective')
    header = fits.Header(make_fitswcs_header(shape, center, scale=[90, 90]*u.arcsec/u.pix,
                                           rotation_angle=17*u.deg))
    grid = Map(np.zeros(shape), header)
    header.update(CRLN_OBS=grid.carrington_longitude.to_value(u.deg),
                  CRLT_OBS=grid.carrington_latitude.to_value(u.deg),
                  OBS_VR=1300., OBS_VW=240., OBS_VN=-120., QUALITY=0, TIMESYS='UTC', T_REC='test-record', T_OBS=Time(epoch).tai.isot+'_TAI')
    wavelengths = {'HMI': {'wavelength_center': 6173.3433*u.AA,
                          'wavelength_grid': np.array([-.2, -.07, .01, .1, .3])*u.AA}}
    config = {'model': {'type': 'mlp', 'dim': 16, 'n_layers': 2, 'encoding_config': {'type': 'identity'}},
              'instrument': [{'type': 'generic', 'instrument_id': 'HMI', 'line_id': 'fe-i-6173'}],
              'stokes_loss': {'type': 'mse', 'continuum_normalization': True},
              'data': {}, 'training': {'max_steps': 2}}
    module = MESphericalModule(image_shape=shape, wavelength_config=wavelengths, gauss_per_dB=1000,
        Rs_per_ds=1., seconds_per_dt=3600., model_config=config['model'],
        instrument_config=config['instrument'], stokes_loss_config=config['stokes_loss'])
    observation = {'id': 'test', 'instrument_id': 'HMI', 'shape': list(shape),
                   'header': dict(header), 'time': {'model_time': epoch}}
    data = SimpleNamespace(image_shape=shape, wavelength_config=wavelengths, gauss_per_dB=1000,
        Rs_per_ds=1., seconds_per_dt=3600., ref_time=datetime(2024, 3, 23, 2),
        times=[datetime(2024, 3, 23, 2)], observations=[observation],
        spectral_response_files={}, spectral_response_by_acquisition={})
    snapshot_path = save_snapshot(module, data, config, tmp_path/'inversion.pme')
    with load(snapshot_path, device='cpu') as snapshot:
        result = snapshot.evaluate(parameters=[*workflow.PARAMETERS, 'B_image_x', 'B_image_y', 'vdop'])
    raw_azimuth = np.rad2deg(np.arctan2(-result.arrays['B_image_x'], result.arrays['B_image_y']))
    arrays = {'field': (result.arrays['B'], 'G'), 'inclination': (result.arrays['inclination'], 'deg'),
              'azimuth': (raw_azimuth % 180, 'deg'), 'vlos_mag': (result.arrays['vdop']*100, 'cm/s'),
              'disambig': (np.where(raw_azimuth < 0, 7, 0).astype(np.uint8), '1'),
              'conf_disambig': (np.full(shape, 90, dtype=np.uint8), '1')}
    directory = tmp_path/'reference'
    directory.mkdir()
    for segment, (values, unit) in arrays.items():
        own = header.copy()
        own['BUNIT'] = unit
        fits.writeto(directory/f'hmi.B_90s.test.{segment}.fits', values, own)
    return snapshot_path, directory, result


def test_transforms_and_observer_correction_match_snapshot(collection):
    snapshot_path, directory, result = collection
    files = workflow.reference_record(directory)
    with load(snapshot_path, device='cpu') as snapshot:
        grid, _, observation = workflow.reference_grid(files, snapshot, resolution=0)
        reference = workflow.sample_reference(files, grid)
        workflow.transform_reference(reference, grid, snapshot, observation, batch_size=11)
    for name in ('Br', 'Btheta', 'Bphi'):
        np.testing.assert_allclose(reference[name], result.arrays[name], atol=.02)
    np.testing.assert_allclose(reference['v_los']*1000, result.arrays['v_los'], atol=.002)


def test_crop_and_record_validation(collection):
    snapshot_path, directory, _ = collection
    files = workflow.reference_record(directory, '2024-03-23T02:00:01')
    with load(snapshot_path, device='cpu') as snapshot:
        grid, _, _ = workflow.reference_grid(files, snapshot, [-200, 200, -200, 200], resolution=4)
    assert max(grid.data.shape) <= 4
    with fits.open(files['inclination'], mode='update') as hdus:
        hdus[0].header['T_OBS'] = Time('2024-03-23T02:00:02').tai.isot+'_TAI'
    with load(snapshot_path, device='cpu') as snapshot:
        with pytest.warns(RuntimeWarning, match='T_OBS differs'):
            workflow.reference_grid(files, snapshot)


def test_missing_segment_rejected(collection):
    _, directory, _ = collection
    next(directory.glob('*.azimuth.fits')).unlink()
    with pytest.raises(ValueError, match='one azimuth'):
        workflow.reference_record(directory)


def test_full_snapshot_to_paper_figures(collection, tmp_path):
    snapshot_path, directory, _ = collection
    for path in directory.glob('*.fits'):
        with fits.open(path, mode='update') as hdus:
            del hdus[0].header['BUNIT']
    with pytest.warns(RuntimeWarning, match='assuming standard HMI') as recorded:
        output = workflow.evaluate(snapshot_path, directory, tmp_path/'figures', resolution=0,
                                   device='cpu', batch_size=31, dpi=90)
    assert len(recorded) == 4
    report = json.loads((output/'comparison.json').read_text())
    assert len(report['warnings']) == 4
    for stem in ('inversion_parameters', 'magnetic_vector', 'los_velocity', 'velocity_vector', 'scatter'):
        for extension in ('pdf', 'png'):
            assert (output/f'{stem}.{extension}').stat().st_size > 1000
    assert (output/'metrics.csv').is_file()
    assert (output/'comparison.json').is_file()
    with np.load(output/'comparison_arrays.npz') as arrays:
        # Float32 network inference can differ slightly with batch shape.
        assert np.nanmax(np.abs(arrays['residual_v_los'])) < 5e-5  # 0.05 m/s
        assert np.nanmax(np.abs(arrays['residual_azimuth'])) < .001


def test_ancillary_metadata_is_optional_and_quality_only_warns(collection):
    snapshot_path, directory, _ = collection
    files = workflow.reference_record(directory)
    for path in files.values():
        with fits.open(path, mode='update') as hdus:
            for key in ('OBS_VR', 'OBS_VW', 'OBS_VN', 'CRLN_OBS', 'CRLT_OBS', 'RSUN_REF'):
                del hdus[0].header[key]
            hdus[0].header['QUALITY'] = 1
    with load(snapshot_path, device='cpu') as snapshot:
        with pytest.warns(RuntimeWarning) as recorded:
            grid, _, observation = workflow.reference_grid(files, snapshot)
        assert any('default to zero' in str(item.message) for item in recorded)
        result = snapshot.evaluate(observation, parameters=['B'])
        assert result.mask.any()
        assert result.metadata['observer_velocity'] == [0., 0., 0.]


def test_explicit_units_override_hmi_defaults(collection):
    snapshot_path, directory, _ = collection
    files = workflow.reference_record(directory)
    with load(snapshot_path, device='cpu') as snapshot:
        grid, _, _ = workflow.reference_grid(files, snapshot)
    expected = workflow.sample_reference(files, grid)
    with fits.open(files['vlos_mag'], mode='update') as hdus:
        hdus[0].data = hdus[0].data/1e5
        hdus[0].header['BUNIT'] = 'km/s'
    actual = workflow.sample_reference(files, grid)
    np.testing.assert_allclose(actual['v_los'], expected['v_los'], atol=1e-5)


def test_azimuth_wrap_and_inclination_endpoints():
    metric, _ = workflow.comparison(np.array([1., 179.]), np.array([179., 1.]), period=180)
    assert metric['mae'] == 2
    assert metric['pearson_r'] is None
    vector = workflow.magnetic_vector(np.array([10., 10.]), np.array([0., 180.]), np.array([0., 0.]))
    np.testing.assert_allclose(vector[:, 2], [10., -10.])


def test_pipeline_disambiguation_bits_and_confidence():
    azimuth = np.full(7, 10.)
    bitmap = np.array([4., 1., 4., 1., 7., 0., 8.])
    confidence = np.array([90., 60., 50., 50., 0., 90., 50.])
    corrected, flipped = workflow.disambiguate_azimuth(azimuth, bitmap, confidence)
    np.testing.assert_allclose(corrected, [10., 190., 190., 10., np.nan, 10., 10.], equal_nan=True)
    np.testing.assert_array_equal(flipped, [False, True, True, False, False, False, False])
    before = workflow.magnetic_vector(np.ones(7), np.full(7, 60.), azimuth)
    after = workflow.magnetic_vector(np.ones(7), np.full(7, 60.), corrected)
    np.testing.assert_allclose(after[flipped, :2], -before[flipped, :2], atol=1e-12)
    np.testing.assert_allclose(after[flipped, 2], before[flipped, 2])


def test_disambiguation_is_not_applied_twice(collection):
    snapshot_path, directory, _ = collection
    files = workflow.reference_record(directory)
    with load(snapshot_path, device='cpu') as snapshot:
        grid, _, _ = workflow.reference_grid(files, snapshot, resolution=0)
    expected = workflow.sample_reference(files, grid)['azimuth_brtp']
    with fits.open(files['azimuth'], mode='update') as hdus:
        hdus[0].data = expected
    actual = workflow.sample_reference(files, grid, azimuth_disambiguated=True)['azimuth_brtp']
    np.testing.assert_allclose(actual, expected)


def test_missing_disambiguation_does_not_invent_signed_vectors(collection):
    snapshot_path, directory, _ = collection
    next(directory.glob('*.disambig.fits')).unlink()
    files = workflow.reference_record(directory)
    with load(snapshot_path, device='cpu') as snapshot:
        grid, _, observation = workflow.reference_grid(files, snapshot, resolution=0)
        with pytest.warns(RuntimeWarning, match='skipping signed Brtp'):
            reference = workflow.sample_reference(files, grid)
        workflow.transform_reference(reference, grid, snapshot, observation)
    assert np.isfinite(reference['B']).all()
    assert np.isnan(reference['Br']).all()


def test_supplemental_disambiguation_collection_is_paired(collection):
    _, directory, _ = collection
    supplement = directory/'disambiguation'
    supplement.mkdir()
    for segment in ('disambig', 'conf_disambig'):
        path = next(directory.glob(f'*.{segment}.fits'))
        path.rename(supplement/path.name)
    files = workflow.reference_record(directory)
    assert files['disambig'].parent == supplement
    assert files['conf_disambig'].parent == supplement
