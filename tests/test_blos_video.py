"""Geometry, cadence, native magnetic sampling and JPEG frame export."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from PIL import Image

from spinnme import load
from test_hmi_reference import collection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
video = importlib.import_module('blos_video')


@pytest.fixture
def references(collection):
    snapshot_path, directory, _ = collection
    for path in list(directory.glob('*.fits')):
        if any('.'+segment+'.' in path.name for segment in ('field', 'inclination', 'azimuth')):
            path.rename(path.with_name(path.name.replace('B_90s', 'B_720s')))
    return snapshot_path, directory


def test_cadence_and_midpoint_ties():
    records = [type('Record', (), {'epoch': Time(f'2024-03-23T02:{minute:02d}:00')}) for minute in (0, 12, 24)]
    times = video.frame_times('2024-03-23T02:00:00', '2024-03-23T02:24:00', 10)
    assert len(times) == 145
    assert video.nearest_indices(records, times)[36] == 0
    assert video.nearest_indices(records, times)[37] == 1
    assert video.nearest_indices(records, times)[108] == 1
    for cadence in (0, -1, np.nan):
        with pytest.raises(ValueError):
            video.frame_times(times[0], times[-1], cadence)


def test_frame_projection_matches_snapshot(references):
    snapshot_path, directory = references
    record = video.index_reference(directory)[0]
    source = video.reference_map(record)
    center = (source.carrington_longitude.to_value(u.deg), source.carrington_latitude.to_value(u.deg))
    frozen = video.surface_patch([record], center, (4, 6))
    assert frozen[0].shape == (4, 6)
    with load(snapshot_path, device='cpu') as snapshot:
        reference, geometry, row, coordinates = video.prepare_reference(record, frozen, snapshot)
        for offset in (0, 10):
            epoch = record.epoch+offset*u.s
            current = video.frame_record([record], epoch)
            geometry, row, coordinates = video.prepare_geometry(current, frozen, snapshot)
            assert abs((Time(row['time']['model_time'])-epoch).to_value(u.s)) < 1e-6
            fast = video.evaluate_magnetic(snapshot, geometry, epoch, snapshot.instrument_id, batch_size=5)
            public = snapshot.sample(coordinates, observation=row, time=epoch,
                                     parameters=['B_los', 'B_transverse', 'azimuth'], batch_size=7)
            for key in ('B_los', 'B_transverse'):
                np.testing.assert_allclose(fast[key], public.arrays[key].ravel(), atol=.001, rtol=1e-5)
            np.testing.assert_allclose(video.azimuth_difference(fast['azimuth'], public.arrays['azimuth'].ravel()),
                                       0, atol=.001)
    assert all(np.isfinite(values).all() for values in reference.values())
    far_side = video.surface_patch([record], ((center[0]+180) % 360, -center[1]), (4, 6))
    with load(snapshot_path, device='cpu') as snapshot:
        hidden, _, _ = video.prepare_geometry(record, far_side, snapshot)
    assert (hidden['mu'] <= 0).all()


def test_native_blos_conversion_precedes_interpolation(references):
    snapshot_path, directory = references
    record = video.index_reference(directory)[0]
    # Opposite polarities with equal |B|. Interpolating inclination first would
    # create a different answer where strengths differ between adjacent pixels.
    with fits.open(record.field, mode='update') as hdus:
        y, x = np.indices(hdus[0].data.shape)
        hdus[0].data[:] = np.where(x % 2, 100., 1000.)
    with fits.open(record.inclination, mode='update') as hdus:
        hdus[0].data[:] = np.where(x % 2, 180., 0.)
    source = video.reference_map(record)
    center = (source.carrington_longitude.to_value(u.deg), source.carrington_latitude.to_value(u.deg))
    frozen = video.surface_patch([record], center, (4, 6))
    with load(snapshot_path, device='cpu') as snapshot:
        sampled, _, _, coordinates = video.prepare_reference(record, frozen, snapshot)
    from scipy.ndimage import map_coordinates
    hpc = coordinates.transform_to(source.coordinate_frame)
    xx, yy = source.wcs.world_to_pixel(hpc)
    expected = map_coordinates(np.where(x % 2, -100., 1000.), [yy, xx], order=1, prefilter=False)
    np.testing.assert_allclose(sampled['B_los'].ravel(), expected, atol=1e-5)


def test_jpeg_frames_and_reference_switch(references, tmp_path, monkeypatch):
    snapshot_path, directory = references
    record = video.index_reference(directory)[0]
    # Two references separated by 20 s; three model frames exercise tie and switch.
    for path in (record.field, record.inclination, record.azimuth):
        with fits.open(path) as hdus:
            header = hdus[0].header.copy()
            header['DATE-OBS'] = (record.epoch+20*u.s).isot
            header['T_OBS'] = (record.epoch+20*u.s).tai.isot+'_TAI'
            header['T_REC'] = 'second-record'
            fits.writeto(path.with_name(path.name.replace('.test.', '.second.')), hdus[0].data, header)
    source = video.reference_map(record)
    center = (source.carrington_longitude.to_value(u.deg), source.carrington_latitude.to_value(u.deg))
    original = video.prepare_reference
    calls = []
    def tracked(*args, **kwargs):
        calls.append(args[0].field)
        return original(*args, **kwargs)
    monkeypatch.setattr(video, 'prepare_reference', tracked)
    model_calls = []
    evaluate = video.evaluate_magnetic
    def evaluated(snapshot, geometry, epoch, *args, **kwargs):
        model_calls.append((Time(epoch).utc.isot, geometry['rtp_to_img_transform'].copy()))
        return evaluate(snapshot, geometry, epoch, *args, **kwargs)
    monkeypatch.setattr(video, 'evaluate_magnetic', evaluated)
    from matplotlib.figure import Figure
    savefig = Figure.savefig
    headings = []
    def save_frame(fig, *args, **kwargs):
        headings.append(fig._suptitle.get_text())
        assert len([axis for axis in fig.axes if axis.images]) == 9
        return savefig(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', save_frame)
    output = tmp_path/'comparison_frames'
    with pytest.warns(RuntimeWarning) as caught:
        video.generate(snapshot_path, directory, output, center, record.epoch, record.epoch+20*u.s,
            shape=(4, 6), cadence=10, device='cpu', dpi=60, max_reference_age=5)
    assert any('outside saved training times' in str(item.message) for item in caught)
    assert any('REFERENCE GAP' in str(item.message) for item in caught)
    assert sorted(path.name for path in output.iterdir()) == [f'frame_{i:06d}.jpg' for i in range(3)]
    for path in output.iterdir():
        with Image.open(path) as image:
            assert image.format == 'JPEG'
            image.verify()
    assert 'REFERENCE GAP' in headings[1]
    assert len(calls) == 2
    assert [epoch for epoch, _ in model_calls] == [(record.epoch+offset*u.s).isot for offset in (0, 10, 20)]
    assert f'HMI {record.epoch.isot} UTC' in headings[1]
    assert f'HMI {(record.epoch+20*u.s).isot} UTC' in headings[2]
    with pytest.raises(FileExistsError, match='new or empty frame directory'):
        video.generate(snapshot_path, directory, output, center, record.epoch, record.epoch)


def test_missing_inclination_rejected(references):
    _, directory = references
    next(directory.glob('*.inclination.fits')).unlink()
    with pytest.raises(ValueError, match='Missing inclination'):
        video.index_reference(directory)


def test_azimuth_wrap_interpolation_and_residual(references):
    _, directory = references
    record = video.index_reference(directory)[0]
    with fits.open(record.azimuth, mode='update') as hdus:
        _, x = np.indices(hdus[0].data.shape)
        hdus[0].data[:] = np.where(x % 2, 1., 179.)
    source = video.reference_map(record)
    coordinate = source.pixel_to_world(np.array([6.5])*u.pix, np.array([6.])*u.pix)
    angle = video.sample_azimuth(record.azimuth, coordinate.Tx.to_value(u.arcsec), coordinate.Ty.to_value(u.arcsec))
    np.testing.assert_allclose(video.azimuth_difference(angle, 0), 0, atol=1e-7)
    np.testing.assert_allclose(video.azimuth_difference(np.array([1., 179.]), np.array([179., 1.])), [2., -2.])


def test_panel_normalizations():
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    styles = video.panel_styles(1500, 300)
    los = styles['B_los'][0]
    assert isinstance(los, SymLogNorm) and los.linthresh == 1
    assert (los.vmin, los.vmax) == (-1500, 1500)
    transverse = styles['B_transverse'][0]
    assert isinstance(transverse, LogNorm) and transverse.vmin == 1 and transverse.vmax == 1500
    azimuth = styles['azimuth'][0]
    assert isinstance(azimuth, Normalize) and (azimuth.vmin, azimuth.vmax) == (0, 180)
    assert (styles['azimuth'][2].vmin, styles['azimuth'][2].vmax) == (-90, 90)


def test_missing_azimuth_rejected(references):
    _, directory = references
    next(directory.glob('*.azimuth.fits')).unlink()
    with pytest.raises(ValueError, match='Missing azimuth'):
        video.index_reference(directory)


def test_continuous_observer_geometry_across_nearest_reference_switch(references):
    from dataclasses import replace
    snapshot_path, directory = references
    original = video.index_reference(directory)[0]
    header = original.header.copy()
    for key in ('HGLN_OBS', 'HGLT_OBS'):
        header.remove(key, ignore_missing=True)
    header['CRLN_OBS'], header['CRLT_OBS'] = 359., -5.
    before = replace(original, header=header)
    header = header.copy()
    header['CRLN_OBS'], header['CRLT_OBS'] = 1., -7.
    header['DSUN_OBS'] *= 1.01
    after = replace(original, header=header, epoch=original.epoch+20*u.s)
    records = [before, after]
    for record in records:
        actual = video.reference_map(video.frame_record(records, record.epoch))
        assert actual.carrington_longitude.deg == pytest.approx(record.header['CRLN_OBS'])
        assert actual.carrington_latitude.deg == pytest.approx(record.header['CRLT_OBS'])
        assert actual.dsun.to_value(u.m) == pytest.approx(record.header['DSUN_OBS'])
        assert actual.scale.axis1.to_value(u.arcsec/u.pix) == pytest.approx(abs(video.reference_map(record).scale.axis1.to_value(u.arcsec/u.pix)))
    middle = video.frame_record(records, original.epoch+10*u.s)
    middle_map = video.reference_map(middle)
    assert abs((middle_map.carrington_longitude.deg+180) % 360-180) < 1e-6
    assert middle_map.carrington_latitude.deg == pytest.approx(-6.)
    assert middle_map.dsun.to_value(u.m) == pytest.approx((before.header['DSUN_OBS']+after.header['DSUN_OBS'])/2)
    center = (0., -6.)
    frozen = video.surface_patch(records, center, (4, 6))
    with load(snapshot_path, device='cpu') as snapshot:
        geometries = []
        for seconds in (9.999, 10.001):
            epoch = original.epoch+seconds*u.s
            current = video.frame_record(records, epoch)
            geometry, row, coordinates = video.prepare_geometry(current, frozen, snapshot)
            geometries.append(geometry)
            assert abs((Time(row['time']['model_time'])-epoch).to_value(u.s)) < 1e-6
            result = video.evaluate_magnetic(snapshot, geometry, epoch, snapshot.instrument_id)
            public = snapshot.sample(coordinates, observation=row, time=epoch,
                                     parameters=['B_los', 'B_transverse', 'azimuth'])
            np.testing.assert_allclose(result['B_los'], public.arrays['B_los'], atol=.001)
        # Nearest HMI changes across this midpoint, but the projection is continuous.
        jump = np.max(abs(geometries[1]['rtp_to_img_transform']-geometries[0]['rtp_to_img_transform']))
        assert 0 < jump < 1e-5
        np.testing.assert_allclose(geometries[0]['coords'][:, :3], geometries[1]['coords'][:, :3], atol=1e-10)
    # No selected epoch changes the surface footprint.
    reordered = video.surface_patch(records[::-1], center, (4, 6))
    for a, b in zip(frozen, reordered):
        np.testing.assert_array_equal(a, b)
