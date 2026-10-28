"""Native Hinode coordinates, HMI reprojection and direct snapshot sampling."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map, make_fitswcs_header

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
workflow = importlib.import_module('hmi_hinode_comparison')


@pytest.fixture
def scene(tmp_path):
    origin = Time('2024-03-23T02:00:00').unix
    y, x = np.meshgrid(np.array([-2., -1., 1., 2.]), np.arange(-2., 3.), indexing='ij')
    x = x+.03*y  # Deliberately not a regular TAN grid.
    hdus = [fits.PrimaryHDU(header=fits.Header({'INVCODE': 'MERLIN'}))]
    for name, data in {
        'Field_Strength': 100+2*x+3*y, 'Field_Inclination': np.zeros(x.shape),
        'Field_Azimuth': (160+10*x) % 180,
        'Stray_Light_Fill_Factor': np.full(x.shape, .8),
        'Original_Continuum_Intensity': np.ones(x.shape),
        'Fitting_Attribute': np.full(x.shape, 4, dtype=np.uint8),
        'X_Coordinate': x, 'Y_Coordinate': y,
    }.items():
        hdus.append(fits.ImageHDU(data, name=name))
    slit_times = list(Time(origin+np.array([30, 60, 90, 120, 150]), format='unix').utc.isot)
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name='TIMES', format='23A', array=slit_times)], name='Times'))
    hinode = tmp_path/'merlin.fits'
    fits.HDUList(hdus).writeto(hinode)
    reference = tmp_path/'scene-reference'
    reference.mkdir()
    shape = (31, 31)
    from _common import _world
    for offset in (0, 90, 180):
        date = Time(origin+offset, format='unix').utc.isot
        center = SkyCoord(0*u.arcsec, 0*u.arcsec, observer='earth', obstime=date, frame='helioprojective')
        header = fits.Header(make_fitswcs_header(shape, center, scale=[1, 1]*u.arcsec/u.pix, rotation_angle=(13+offset/90*7)*u.deg))
        grid = Map(np.zeros(shape), header)
        header.update(CRLN_OBS=grid.carrington_longitude.to_value(u.deg),
                      CRLT_OBS=grid.carrington_latitude.to_value(u.deg), QUALITY=0, T_REC=str(offset), TIMESYS='UTC', T_OBS=Time(date).tai.isot+'_TAI')
        hx, hy = _world(header, shape)
        for segment, data in [('field', 100+2*hx+3*hy+offset), ('inclination', np.zeros(shape)), ('azimuth', (160+10*hx) % 180)]:
            fits.writeto(reference/f'hmi.B_720s.{offset}.{segment}.fits', data, header)
    return reference, hinode, x, y


def test_time_scale_and_nearest_pairing():
    begin, finish = workflow.time_window(workflow.START, workflow.END, 'tai')
    assert Time(begin, format='unix').utc.isot == '2024-03-23T00:59:23.000'
    assert Time(finish, format='unix').utc.isot == '2024-03-23T02:59:23.000'
    with pytest.raises(ValueError, match='later'):
        workflow.time_window(workflow.END, workflow.START, 'utc')
    origin = Time('2024-03-23T02:00:00').unix
    records = [{'time': Time(origin+offset, format='unix').isot} for offset in (0, 90, 270)]
    times = origin+np.array([[-1, 0, 44, 45, 46, 135, 180, 269, 271, np.nan]])
    indices, offsets, valid = workflow.pair_times(times, records, maximum_offset=45)
    np.testing.assert_array_equal(indices[0, :5], [0, 0, 0, 0, 1])
    np.testing.assert_allclose(offsets[0, 2:5], [-44, -45, 44])
    np.testing.assert_array_equal(valid, [[False, True, True, True, True, True, False, True, False, False]])


def test_native_crop_and_optional_quality_filter(scene):
    _, hinode, x, y = scene
    source = workflow.merlin({**workflow.HINODE, 'file': str(hinode)})
    begin, finish = workflow.time_window('2024-03-23T02:01:00', '2024-03-23T02:02:00', 'utc')
    cropped = workflow.native_crop(source, begin, finish, [-1.5, 1.5, -1.5, 1.5])
    np.testing.assert_array_equal(cropped['x'], x[1:3, 1:4])
    assert cropped['mask'].all()
    rejected = workflow.merlin({**workflow.HINODE, 'file': str(hinode), 'accepted_fit_status': [3]})
    assert not rejected['mask'].any()


from test_hmi_reference import collection



def test_one_combined_comparison_plot_uses_exact_coordinates(scene, tmp_path, monkeypatch):
    _, _, x, y = scene
    arrays = {'x_arcsec': 40*x-180, 'y_arcsec': 30*y}
    save_figure = workflow.save_figure
    def check_limits(fig, *args, **kwargs):
        panels = [ax for ax in fig.axes if ax.get_xlabel() == 'Solar X [arcsec]']
        assert len(panels) == 4
        assert all(ax.get_xlim()[1] == -110. for ax in panels)
        for ax in panels:
            assert ax.collections[0].cmap.name == 'infinity'
            assert ax.collections[0].get_alpha() is None
        assert fig.get_constrained_layout()
        distributions = [ax for ax in fig.axes if '≤ value' in ax.get_ylabel()]
        assert len(distributions) == 3
        assert [ax.get_xscale() for ax in distributions] == ['symlog', 'symlog', 'linear']
        assert all(ax.get_xlim()[0] == 0 for ax in distributions[:2])
        for ax in distributions[:2]:
            np.testing.assert_array_equal(ax.get_xticks()[:4], [0, 1, 10, 100])
            assert [tick.get_text() for tick in ax.get_xticklabels()[:4]] == ['0', '$10^{0}$', '$10^{1}$', '$10^{2}$']
        for ax in distributions:
            assert len(ax.lines) == 4
            assert {line.get_linestyle() for line in ax.lines} == {'-', '--', ':', '-.'}
            for line in ax.lines:
                values, cumulative = line.get_data()
                assert cumulative[0] == 0 and cumulative[-1] == 1
                assert np.all(np.diff(cumulative) >= 0)
                assert np.all(np.diff(values) >= 0)
        # Two visible columns carry weights 0.5 and 1 per row. Their values
        # alternate in sorted order; the weaker columns contribute no weight.
        azimuth = distributions[-1].lines[0]
        np.testing.assert_allclose(azimuth.get_ydata(),
                                   np.array([0, 0, 1, 3, 4, 6, 7, 9, 10, 12, 12])/12)
        return save_figure(fig, *args, **kwargs)
    monkeypatch.setattr(workflow, 'save_figure', check_limits)
    for source in workflow.SOURCES:
        for parameter in workflow.PARAMETERS:
            arrays[source+'_'+parameter] = 100+2*x+3*y
    for source in workflow.SOURCES:
        arrays[source+'_B'] = np.broadcast_to([0., 10., 55., 100., 150.], x.shape)
    output = tmp_path/'figures'
    output.mkdir()
    workflow._figures(output, arrays)
    assert sorted(path.name for path in output.iterdir()) == ['comparison.pdf', 'comparison.png']



def test_720_second_reference_pairing():
    origin = Time('2024-03-23T02:00:00').unix
    records = [{'time': Time(origin+offset, format='unix').isot} for offset in (0, 720, 1440)]
    times = origin+np.array([[-1, 0, 359, 360, 361, 1080, 1440, 1441]])
    indices, offsets, valid = workflow.pair_times(times, records)
    np.testing.assert_array_equal(indices, [[0, 0, 0, 0, 1, 1, 2, 2]])
    np.testing.assert_allclose(offsets, [[1, 0, -359, -360, 359, -360, 0, -1]])
    np.testing.assert_array_equal(valid, [[False, True, True, True, True, True, True, False]])
    assert workflow.MAXIMUM_TIME_OFFSET_SECONDS == 360
    assert workflow.HMI_REFERENCE.endswith('_720s')



def test_plot_preserves_independent_native_hinode_pixels(tmp_path, monkeypatch):
    x, y = np.meshgrid(np.linspace(-320, -110, 5), np.linspace(-150, -10, 4))
    nx, ny = np.meshgrid(np.linspace(-300, -130, 7), np.linspace(-140, -20, 9))
    arrays = {'x_arcsec': x, 'y_arcsec': y,
              'spinnme_x_arcsec': x-25., 'spinnme_y_arcsec': y-30.,
              'spinnme_hinode_x_arcsec': nx, 'spinnme_hinode_y_arcsec': ny}
    for source in workflow.SOURCES:
        shape = nx.shape if source == 'spinnme_hinode' else x.shape
        for parameter in workflow.PARAMETERS:
            arrays[source+'_'+parameter] = np.arange(np.prod(shape)).reshape(shape).astype(float)
        arrays[source+'_B'] = np.full(shape, 200.)
    save = workflow.save_figure
    def inspect(fig, *args, **kwargs):
        panels = [ax for ax in fig.axes if ax.get_xlabel() == 'Solar X [arcsec]']
        hmi_vertices = panels[1].collections[0].get_coordinates()
        for ax in panels:
            assert ax.get_xlim() == (-345., -110.)
            assert ax.get_ylim() == (-180., -10.)
        hmi_centers = (hmi_vertices[:-1, :-1]+hmi_vertices[1:, :-1]+
                       hmi_vertices[:-1, 1:]+hmi_vertices[1:, 1:])/4
        np.testing.assert_allclose(hmi_centers[..., 0], x-25.)
        np.testing.assert_allclose(hmi_centers[..., 1], y-30.)
        native = panels[3].collections[0]
        assert native.get_array().shape == nx.shape
        assert np.ma.count(native.get_array()) == nx.size
        vertices = native.get_coordinates()
        centers = (vertices[:-1, :-1]+vertices[1:, :-1]+vertices[:-1, 1:]+vertices[1:, 1:])/4
        np.testing.assert_allclose(centers[..., 0], nx)
        np.testing.assert_allclose(centers[..., 1], ny)
        distributions = [ax for ax in fig.axes if '≤ value' in ax.get_ylabel()]
        assert len(distributions[0].lines[3].get_xdata()) > len(distributions[0].lines[0].get_xdata())
        return save(fig, *args, **kwargs)
    monkeypatch.setattr(workflow, 'save_figure', inspect)
    workflow._figures(tmp_path, arrays, dpi=100)


@pytest.fixture
def saved_hinode(collection, tmp_path):
    """A portable native grid with different scales, slit times, rolls and shape."""
    import base64
    import torch
    from datetime import datetime
    from spinnme.snapshots import decode, encode
    from sunpy.coordinates import frames
    from sunpy.coordinates.ephemeris import get_earth
    source_path, _, _ = collection
    payload = torch.load(source_path, weights_only=True)
    metadata = decode(json.loads(payload['metadata_json']))
    metadata['init']['Rs_per_ds'] = 2.5
    metadata['init']['seconds_per_dt'] = 1800.
    metadata['ref_time'] = datetime(2024, 3, 23, 1, 58)
    row = metadata['observations'][0]
    row.update(id='native-hinode', shape=[3, 4], role='train')
    row['time'] = {'model_time': '2024-03-23T02:00:00'}
    offsets = np.array([31.125, 61.25, 91.5, 121.75])
    epochs = Time(row['time']['model_time'])+offsets*u.s
    observers = get_earth(epochs)
    central = observers.transform_to(frames.HeliographicCarrington(observer=observers, obstime=epochs))
    row['slit_pixel_world'] = [[-1.8, -.7, .4, 1.6], [-1.2, -1.1, -.9, -.8], [.03]*4, [.8]*4]
    row['slit_geometry'] = {'obs_lon': central.lon.to_value(u.rad).tolist(),
                            'obs_lat': central.lat.to_value(u.rad).tolist(),
                            'pAng': [.1, .15, -.2, -.1],
                            'observer_distance': observers.radius.to_value(u.Rsun).tolist(),
                            'observer_velocity': np.zeros((4, 3)).tolist()}
    row['time_offsets_seconds'], row['time_offset_axis'] = offsets.tolist(), 'x'
    mask = np.ones((3, 4), bool)
    mask[0, 0] = False
    row['valid_mask_bits'] = base64.b64encode(np.packbits(mask).tobytes()).decode()
    payload['metadata_json'] = json.dumps(encode(metadata))
    result = tmp_path/'hinode.pme'
    torch.save(payload, result)
    return result


def test_saved_native_grid_and_projection_match_source_inversion(saved_hinode, tmp_path):
    import spinnme
    inversion = workflow.read_hinode_inversion(saved_hinode)
    with spinnme.load(saved_hinode, device='cpu') as model:
        geometry = model.geometry('native-hinode')
        expected = model.evaluate('native-hinode', parameters=['B', 'B_los', 'B_transverse', 'azimuth'])
        actual = workflow.sample_snapshot(model, inversion, 5, image_roll=inversion['image_p_angle'])
        np.testing.assert_array_equal(inversion['coords'].reshape(-1, 4), geometry['coords'])
        np.testing.assert_array_equal(actual.mask, expected.mask)
        for key in expected.arrays:
            np.testing.assert_allclose(actual.arrays[key], expected.arrays[key], rtol=1e-5, atol=.002, equal_nan=True)
        # Existing FITS arguments resolve their snapshot; modified display HDUs
        # never become replacement model coordinates.
        directory = tmp_path/'fits'
        expected.write_fits(directory)
    with fits.open(directory/'B.fits', mode='update') as hdus:
        hdus['X_COORDINATE'].data += 100.
    restored = workflow.read_hinode_inversion(directory)
    np.testing.assert_array_equal(restored['coords'], inversion['coords'])
    np.testing.assert_array_equal(restored['times'], inversion['times'])


def test_target_receives_exact_xyzt_with_its_own_normalization(saved_hinode, collection, monkeypatch):
    import spinnme
    from spinnme import progress
    inversion = workflow.read_hinode_inversion(saved_hinode)
    calls, bars = [], []
    original_progress = progress.progress
    monkeypatch.setattr(progress, 'progress', lambda iterable, **kwargs: (bars.append(kwargs) or original_progress(iterable, **kwargs)))
    with spinnme.load(collection[0], device='cpu') as target:
        original_forward = target.parameter_model.forward
        def forward(coords, **kwargs):
            calls.append(coords.detach().cpu().numpy().copy())
            return original_forward(coords, **kwargs)
        monkeypatch.setattr(target.parameter_model, 'forward', forward)
        actual = workflow.sample_snapshot(target, inversion, 5)
        expected = inversion['coords'][inversion['mask']].copy()
        expected[:, :3] *= inversion['Rs_per_ds']/target.Rs_per_ds
        expected[:, 3] = (expected[:, 3]*inversion['seconds_per_dt'] +
                          (Time(inversion['ref_time'])-Time(target.ref_time)).to_value(u.s))/target.seconds_per_dt
    np.testing.assert_array_equal(np.concatenate(calls), expected.astype(np.float32))
    assert len(bars) == 1 and bars[0]['desc'] == 'Sampling SPINNME'
    np.testing.assert_array_equal(actual.mask, inversion['mask'])


@pytest.mark.parametrize('shift', [(0., 0.), (2., 3.)])
def test_comparison_samples_inversion_grid_once_independent_of_merlin(scene, saved_hinode, collection, tmp_path, monkeypatch, shift):
    reference, merlin, x, y = scene
    calls, data = [], {}
    sample = workflow.sample_snapshot
    def track(model, inversion, *args, **kwargs):
        calls.append(inversion['coords'].copy())
        return sample(model, inversion, *args, **kwargs)
    monkeypatch.setattr(workflow, 'sample_snapshot', track)
    def plot(output, arrays, *args, **kwargs):
        data.update(arrays)
        (output/'comparison.png').touch()
    monkeypatch.setattr(workflow, '_figures', plot)
    native = workflow.read_hinode_inversion(saved_hinode, shift=shift)
    result = workflow.compare(tmp_path/'comparison', snapshot=collection[0], hinode_result=saved_hinode,
                              reference=reference, hinode=merlin, shift_x=shift[0], shift_y=shift[1],
                              start='2024-03-23T02:00:00', end='2024-03-23T02:03:00', time_scale='utc',
                              batch_size=5, device='cpu')
    assert result.exists() and len(calls) == 1
    np.testing.assert_array_equal(calls[0], native['coords'])
    assert data['spinnme_B'].shape == data['spinnme_hinode_B'].shape == (3, 4)
    assert data['hinode_B'].shape == (4, 5)
    assert data['hmi_B'].shape == (3, 4)
    np.testing.assert_array_equal(data['spinnme_x_arcsec'], native['x'])
    np.testing.assert_array_equal(data['spinnme_y_arcsec'], native['y'])
    np.testing.assert_allclose(data['spinnme_hinode_x_arcsec'], native['x'])
    np.testing.assert_allclose(data['spinnme_hinode_y_arcsec'], native['y'])
    np.testing.assert_array_equal(np.isfinite(data['spinnme_B']), native['mask'])
    expected = 100+2*native['x']+3*native['y']+[0, 90, 90, 90]
    np.testing.assert_allclose(data['hmi_B'], np.where(native['mask'], expected, np.nan), atol=1e-5)
    np.testing.assert_array_equal(data['hmi_x_arcsec'], data['spinnme_x_arcsec'])
    np.testing.assert_array_equal(data['hmi_y_arcsec'], data['spinnme_y_arcsec'])


def test_crop_keeps_exact_native_coordinates_and_projection(saved_hinode):
    native = workflow.read_hinode_inversion(saved_hinode)
    first, last = Time(['2024-03-23T02:00:50', '2024-03-23T02:01:40']).unix
    cropped = workflow.native_crop(native, first, last)
    for key in ('coords', 'cartesian_to_spherical_transform', 'rtp_to_img_transform', 'times'):
        np.testing.assert_array_equal(cropped[key], native[key][:, 1:3])


def test_shifted_comparison_crops_all_predictions_on_common_grid(scene, saved_hinode, collection, tmp_path, monkeypatch):
    reference, merlin, _, _ = scene
    data, calls = {}, []
    sample = workflow.sample_snapshot
    def track(model, inversion, *args, **kwargs):
        calls.append(inversion['coords'].copy())
        return sample(model, inversion, *args, **kwargs)
    monkeypatch.setattr(workflow, 'sample_snapshot', track)
    monkeypatch.setattr(workflow, '_figures', lambda output, arrays, *args, **kwargs: data.update(arrays))
    native = workflow.read_hinode_inversion(saved_hinode, shift=(2., 1.))
    bounds = [-1., 3., -2., 4.]
    begin, finish = workflow.time_window('2024-03-23T02:00:00', '2024-03-23T02:03:00', 'utc')
    expected = workflow.native_crop(native, begin, finish, bounds)
    workflow.compare(tmp_path/'cropped', snapshot=collection[0], hinode_result=saved_hinode,
                     reference=reference, hinode=merlin, shift_x=2., shift_y=1., bounds=bounds,
                     start='2024-03-23T02:00:00', end='2024-03-23T02:03:00', time_scale='utc',
                     batch_size=5, device='cpu')
    np.testing.assert_array_equal(calls[0], expected['coords'])
    np.testing.assert_array_equal(data['spinnme_x_arcsec'], expected['x'])
    np.testing.assert_array_equal(data['spinnme_y_arcsec'], expected['y'])
    np.testing.assert_array_equal(np.isfinite(data['spinnme_B']), expected['mask'])
    for source in ('hmi', 'spinnme_hinode'):
        np.testing.assert_array_equal(data[source+'_x_arcsec'], expected['x'])
        np.testing.assert_array_equal(data[source+'_y_arcsec'], expected['y'])


def test_alignment_does_not_mutate_saved_geometry(saved_hinode):
    native = workflow.read_hinode_inversion(saved_hinode)
    original = {key: value.copy() for key, value in native.items() if isinstance(value, np.ndarray)}
    begin, finish = native['times'].min(), native['times'].max()
    shifted = workflow.aligned_crop(native, begin, finish, None, (25., 30.))
    for key, value in original.items():
        np.testing.assert_array_equal(native[key], value)
    np.testing.assert_array_equal(shifted['coords'], native['coords'])
    np.testing.assert_allclose(shifted['x'], native['x']+25.)
    shifted['mask'][:] = False
    np.testing.assert_array_equal(native['mask'], original['mask'])


def test_comparison_keeps_bracketing_reference_records(scene, saved_hinode, collection, tmp_path, monkeypatch):
    reference, merlin, _, _ = scene
    data = {}
    monkeypatch.setattr(workflow, '_figures', lambda output, arrays, *args, **kwargs: data.update(arrays))
    # Neither HMI record at 00 or 90 seconds is inside this selection. Both
    # remain available to bracket the selected slit at 60 seconds.
    workflow.compare(tmp_path/'edge-window', snapshot=collection[0], hinode_result=saved_hinode,
                     reference=reference, hinode=merlin,
                     start='2024-03-23T02:00:40', end='2024-03-23T02:01:10', time_scale='utc',
                     device='cpu')
    assert np.isfinite(data['hmi_B']).all()
    assert data['hmi_B'].shape == (3, 1)
    assert data['spinnme_B'].shape == (3, 1)


def test_explicit_bounds_control_maps_and_distribution(tmp_path, monkeypatch):
    x, y = np.meshgrid([-200., -150., -100.], [-30., 0., 30.])
    arrays = {'x_arcsec': x, 'y_arcsec': y}
    for source in workflow.SOURCES:
        for parameter in workflow.PARAMETERS:
            arrays[source+'_'+parameter] = np.broadcast_to([-20., 0., 10.], x.shape).copy()
        arrays[source+'_B'] = np.full(x.shape, 100.)
        arrays[source+'_B_transverse'] = np.abs(arrays[source+'_B_transverse'])
    def inspect(fig, *args, **kwargs):
        panels = [ax for ax in fig.axes if ax.get_xlabel() == 'Solar X [arcsec]']
        for ax in panels:
            assert ax.get_xlim() == (-175., -75.)
            assert ax.get_ylim() == (-40., 40.)
        distributions = [ax for ax in fig.axes if '≤ value' in ax.get_ylabel()]
        for line in distributions[0].lines:
            values, fractions = line.get_data()
            # The -200 column is outside the explicit bounds. Zero pixels
            # contribute half the mass; the +10 column must remain visible.
            assert not np.any(values == 20.)
            assert np.any((values == 0.) & (fractions == .5))
            assert np.any((values == 10.) & (fractions == 1.))
    monkeypatch.setattr(workflow, 'save_figure', inspect)
    workflow._figures(tmp_path, arrays, bounds=(-175., -75., -40., 40.))


def test_nonfinite_prediction_is_excluded_from_mask(saved_hinode, collection, monkeypatch):
    import spinnme
    native = workflow.read_hinode_inversion(saved_hinode)
    with spinnme.load(collection[0], device='cpu') as model:
        original = model.parameter_model.forward
        def forward(coords, **kwargs):
            result = original(coords, **kwargs)
            result['b_x'] = result['b_x'] * float('nan')
            return result
        monkeypatch.setattr(model.parameter_model, 'forward', forward)
        result = workflow.sample_snapshot(model, native, 100)
    assert not result.mask.any()


def test_shift_precedes_physical_geometry_and_preserves_slit_times(saved_hinode):
    import spinnme
    original = workflow.read_hinode_inversion(saved_hinode)
    shifted = workflow.read_hinode_inversion(saved_hinode, shift=(25., 30.))
    np.testing.assert_allclose(shifted['x'], original['x']+25.)
    np.testing.assert_allclose(shifted['y'], original['y']+30.)
    np.testing.assert_array_equal(shifted['times'], original['times'])
    np.testing.assert_array_equal(shifted['coords'][..., 3], original['coords'][..., 3])
    assert not np.allclose(shifted['coords'][..., :3], original['coords'][..., :3])
    for key in workflow.PARAMETERS:
        np.testing.assert_array_equal(shifted[key], original[key])
    with spinnme.load(saved_hinode, device='cpu') as model:
        row = model.grid('native-hinode')
        positions = np.asarray(row['slit_pixel_world'])
        positions[0] += 25.
        positions[1] += 30.
        row['slit_pixel_world'] = positions.tolist()
        geometry = model.geometry(row)
        expected = model.evaluate('native-hinode', parameters=['B', 'B_los', 'B_transverse'])
        np.testing.assert_array_equal(shifted['coords'].reshape(-1, 4), geometry['coords'])
        for key in ('cartesian_to_spherical_transform', 'rtp_to_img_transform'):
            np.testing.assert_array_equal(shifted[key].reshape(-1, 3, 3), geometry[key])
        np.testing.assert_allclose(shifted['apparent_B_los'], expected.arrays['B_los'], equal_nan=True)
    restored = workflow.read_hinode_inversion(saved_hinode)
    np.testing.assert_array_equal(restored['coords'], original['coords'])
