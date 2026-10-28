"""Check the two publication plates with a portable test snapshot and rotated WCS."""
import importlib
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import pytest

from test_hmi_reference import collection
from spinnme.exports import read_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
workflow = importlib.import_module('hmi_cadence_comparison')


def test_layout_scales_and_exact_hpc_box(collection):
    _, _, result = collection
    arrays = {name: result.arrays[name] for name in workflow.PARAMETERS}
    scales = workflow.color_scales(3000)
    assert (scales[0][0].vmin, scales[0][0].vmax) == (1, 3000)
    fig = workflow.publication_figure([arrays]*4, result.header, (-1000, 1000, -1000, 1000),
                                     scales, titles=True, zoom_box=workflow.ZOOM_HPC)
    assert fig.get_constrained_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bars = [ax for ax in fig.axes if ax.get_subplotspec().colspan.start == 4]
    for upper, lower in zip(bars, bars[1:]):
        assert not upper.get_tightbbox(renderer).overlaps(lower.get_tightbbox(renderer))
    assert len(fig.axes) == 15
    panels = [ax for ax in fig.axes if ax.patches]
    assert len(panels) == 12
    assert [ax.get_title() for ax in panels[:4]] == list(workflow.TITLES)
    for i, ax in enumerate(panels):
        box = ax.patches[0]
        assert (box.get_x(), box.get_y(), box.get_width(), box.get_height()) == (0, 0, 100, 50)
        assert bool(ax.get_xlabel()) == (i//4 == 2)
        assert bool(ax.get_ylabel()) == (i % 4 == 0)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
    plt.close(fig)
    fig = workflow.publication_figure([arrays]*4, result.header, workflow.ZOOM_HPC, scales, frame=True)
    assert all(not ax.get_title() for ax in fig.axes)
    for ax in fig.axes:
        if ax.get_subplotspec().colspan.start < 4:
            assert ax.spines['top'].get_visible()
            assert ax.spines['right'].get_visible()
    plt.close(fig)


def test_end_to_end_two_references_and_native_zoom(collection, tmp_path):
    snapshot, ref90, _ = collection
    ref720 = tmp_path/'reference720'
    ref720.mkdir()
    for path in ref90.glob('*.fits'):
        shutil.copy(path, ref720/path.name.replace('B_90s', 'B_720s'))
    output = workflow.evaluate(snapshot, snapshot, next(ref90.glob('*.field.fits')),
                               next(ref720.glob('*.field.fits')), tmp_path/'publication',
                               time='2024-03-23T02:00:00', resolution=8, device='cpu', dpi=60)
    assert not list(output.glob('*.json'))
    assert max(read_npz(output/'full_disk_1.npz').mask.shape) == 8
    assert len(list(output.glob('*.npz'))) == 8
    for plate in ('full_disk', 'zoom'):
        assert (output/f'cadence_{plate}.png').stat().st_size > 1000
    with pytest.raises(ValueError, match='positive extent'):
        workflow.evaluate(snapshot, snapshot, ref90, ref720, output, zoom=(0, 0, 0, 50))
    with pytest.raises(ValueError, match='half a cadence'):
        workflow.reference_record(ref90, 90, '2024-03-24T02:00:00')
