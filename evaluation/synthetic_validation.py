"""Compare every synthetic inversion on the same dense, instantaneous truth grids.

Training observations can average finite exposures.  Evaluation always uses the
latent instantaneous truth, so temporal averaging and sparse sampling remain part
of the measured error.  Pixel and frame counts are descriptive sample counts;
each cadence/noise case is evaluated once.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import _broadcast_mask, comparison, read_npz, checksum, utc_now, write_json, save_figure

DEFAULT_PARAMETERS = ["B", "B_los", "B_transverse", "I", "Q", "U", "V"]
DEVICE = None
BATCH_SIZE = 4096
FRAME_STRIDE = 1
FACTORS = ("cadence_s", "exposure_s", "phase_s", "noise_level", "noise_sigma", "noise_std")
METRICS = ("bias", "mae", "rmse", "temporal_mean_rmse", "temporal_mean_bias",
           "temporal_derivative_rmse", "temporal_correlation", "amplitude_ratio")


def _resolve(base: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else base / value


def _time_seconds(values: list[str]) -> np.ndarray:
    from astropy.time import Time
    return np.asarray(Time(values, scale="utc").unix, dtype=float)


def _header(value: Any) -> dict:
    if isinstance(value, str):
        from astropy.io import fits
        value = fits.Header.fromstring(value, sep="\n")
    if not hasattr(value, "items"):
        raise ValueError("Dense truth requires its native FITS WCS header.")
    return {str(key).upper(): val for key, val in value.items()}


def _checked_metadata(result: Any, truth: dict, epoch: str, parameters: list[str]) -> dict:
    """Require comparable epochs, units, observer geometry, and spectral sampling."""
    predicted = {**result.metadata, "header": _header(result.header), "units": result.units}
    reference = {**truth, "header": _header(truth.get("header"))}
    required = ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
                "DSUN_OBS", "RSUN_REF")
    for metadata in (predicted, reference):
        if any(key not in metadata["header"] for key in required):
            raise ValueError("Dense comparisons require complete celestial WCS and solar observer metadata.")
        if not metadata.get("model_time"):
            raise ValueError("Dense comparisons require an explicit model_time for prediction and truth.")
        epochs = _time_seconds([metadata["model_time"], epoch])
        if abs(epochs[0] - epochs[1]) > 1e-4:
            raise ValueError("Dense truth/prediction model epoch differs from its evaluation frame.")
        # Normalize equivalent ISO spellings before using the shared exact-grid guard.
        metadata["model_time"] = epoch
    _check_wcs(predicted["header"], reference["header"], tuple(result.mask.shape))
    if any(name in {"I", "Q", "U", "V", "stokes"} for name in parameters):
        for key in ("wavelength_grid_m", "wavelength_center_m"):
            if key not in predicted or key not in reference:
                raise ValueError("Spectral comparisons require explicit wavelength grid and center metadata.")
            a, b = np.asarray(predicted[key]), np.asarray(reference[key])
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-18):
                raise ValueError(f"Spectral truth and prediction differ in {key}.")
    return reference


def _check_wcs(predicted: dict, truth: dict, shape: tuple[int, int]) -> None:
    """Compare world coordinates, allowing equivalent arcsec/degree WCS headers."""
    from astropy import units as u
    from astropy.wcs import WCS
    a, b = WCS(predicted).celestial, WCS(truth).celestial
    if a.world_axis_physical_types != b.world_axis_physical_types:
        raise ValueError("Dense prediction and truth have incompatible celestial coordinates.")
    y, x = np.meshgrid(np.linspace(0, shape[0]-1, 3), np.linspace(0, shape[1]-1, 3))
    worlds_a, worlds_b = a.pixel_to_world_values(x.ravel(), y.ravel()), b.pixel_to_world_values(x.ravel(), y.ravel())
    for left, right, left_unit, right_unit in zip(worlds_a, worlds_b, a.world_axis_units, b.world_axis_units):
        left = (np.asarray(left) * u.Unit(left_unit)).to_value(u.deg)
        right = (np.asarray(right) * u.Unit(right_unit)).to_value(u.deg)
        delta = (left - right + 180) % 360 - 180
        if not np.isfinite(delta).all() or not np.allclose(delta, 0, rtol=0, atol=1e-9):
            raise ValueError("Dense prediction and truth have different celestial WCS grids.")
    for key in ("DSUN_OBS", "RSUN_REF", "CRLN_OBS", "CRLT_OBS", "HGLN_OBS", "HGLT_OBS"):
        if key in predicted or key in truth:
            if key not in predicted or key not in truth or not np.isclose(predicted[key], truth[key], rtol=1e-10, atol=1e-9):
                raise ValueError(f"Dense prediction and truth differ in observer geometry {key}.")
    for key in ("DATE-OBS", "DATE-AVG"):
        if key in predicted and key in truth:
            epochs = _time_seconds([predicted[key], truth[key]])
            if abs(epochs[0]-epochs[1]) > 1e-4:
                raise ValueError(f"Dense prediction and truth differ in geometry epoch {key}.")


def _unit_factor(source: str | None, target: str | None, name: str) -> float:
    if source is None or target is None:
        raise ValueError(f"Missing explicit unit for {name} in dense truth or prediction.")
    from astropy import units as u
    try:
        return float((1 * u.Unit(source)).to_value(u.Unit(target)))
    except (ValueError, u.UnitConversionError) as error:
        raise ValueError(f"Incompatible units for {name}: {source!r} and {target!r}.") from error


def _subsets(seconds: np.ndarray, observed_seconds: np.ndarray) -> dict[str, np.ndarray]:
    if not len(observed_seconds) or not np.isfinite(observed_seconds).all():
        raise ValueError("Every run needs finite observed acquisition-center times.")
    observed_seconds = np.sort(np.unique(observed_seconds))
    observed = np.any(np.isclose(seconds[:, None], observed_seconds[None], rtol=0, atol=1e-3), axis=1)
    inside = (seconds > observed_seconds[0] + 1e-3) & (seconds < observed_seconds[-1] - 1e-3)
    return {"dense": np.ones(len(seconds), dtype=bool), "observed": observed,
            "held_out": ~observed, "interpolation": ~observed & inside,
            "extrapolation": ~observed & ~inside,
            "boundary": np.isclose(seconds, observed_seconds[0], rtol=0, atol=1e-3)
                        | np.isclose(seconds, observed_seconds[-1], rtol=0, atol=1e-3)}


def _temporal_statistics(rows: list[dict], times: np.ndarray, selected: np.ndarray) -> dict:
    frame_rows = [row for row, use in zip(rows, selected) if use]
    x = times[selected]
    predicted = np.asarray([row["mean_prediction"] for row in frame_rows])
    truth = np.asarray([row["mean_truth"] for row in frame_rows])
    difference = predicted - truth
    correlation, amplitude, derivative = None, None, None
    if len(x) > 1:
        derivative = float(np.sqrt(np.mean((np.diff(difference) / np.diff(x)) ** 2)))
        if np.std(truth) > 0:
            amplitude = float(np.std(predicted) / np.std(truth))
            if np.std(predicted) > 0:
                correlation = float(np.corrcoef(predicted, truth)[0, 1])
    return {"n_frames": len(frame_rows), "n_paired_samples": sum(row["n"] for row in frame_rows),
            "bias": float(np.mean([row["bias"] for row in frame_rows])),
            "mae": float(np.mean([row["mae"] for row in frame_rows])),
            "rmse": float(np.sqrt(np.mean([row["rmse"] ** 2 for row in frame_rows]))),
            "temporal_mean_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "temporal_mean_bias": float(np.mean(difference)),
            "temporal_derivative_rmse": derivative, "temporal_correlation": correlation,
            "amplitude_ratio": amplitude}


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No synthetic evaluation rows were produced.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    """Report each run's temporal metrics without replicate/error-bar machinery."""
    summary = []
    for row in rows:
        for metric in METRICS:
            summary.append({**{key: row[key] for key in ("run_id", *FACTORS, "parameter", "unit", "subset")},
                            "metric": metric, "mean": row[metric]})
    return summary


def _inputs(dataset_path: Path, runs_directory: Path, run_ids=None) -> dict:
    """Match each run to its single generated cadence/noise stream."""
    dataset = json.loads(dataset_path.read_text())
    index = {acquisition['name']: acquisition for acquisition in dataset['acquisitions']}
    rows = [{'run_id': f'cadence{cadence}s-{noise}',
             'acquisition': f'cadence{cadence}s-phase0s-{noise}',
             'snapshot': f'cadence{cadence}s-{noise}_v01/inversion.pme'}
            for cadence in (5, 30, 90, 180, 720)
            for noise in ('noiseless', 'optimistic', 'nominal', 'challenging')]
    if run_ids is not None:
        missing = set(run_ids)-{row['run_id'] for row in rows}
        if missing:
            raise ValueError('Unknown synthetic run IDs: ' + ', '.join(sorted(missing)))
        rows = [row for row in rows if row['run_id'] in run_ids]
    runs = []
    for row in rows:
        acquisition = index[row['acquisition']]
        if len(acquisition['realizations']) != 1:
            raise ValueError('Each evaluation run requires one generated observation stream.')
        observation = acquisition['realizations'][0]
        snapshot = _resolve(runs_directory, row['snapshot']).resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(f'Missing inversion snapshot: {snapshot}')
        sigma = np.asarray(acquisition.get('effective_noise_std', acquisition['noise_std']), dtype=float)
        runs.append({'run_id': row['run_id'], 'snapshot': str(snapshot),
                     'cadence_s': acquisition['cadence_seconds'], 'exposure_s': acquisition['exposure_seconds'],
                     'phase_s': acquisition.get('phase_seconds', 0),
                     'noise_label': acquisition.get('noise_label', acquisition['name']),
                     'noise_std': sigma.tolist(), 'noise_sigma': float(np.max(sigma[1:])),
                     'observed_times': [frame['time'] for frame in observation['frames']]})
    return {'dense_truth': dataset['dense_truth'], 'runs': runs}


def evaluate_series(dataset: str | Path, runs_directory: str | Path, output: str | Path, *, run_ids: list[str] | None = None,
                    parameters: list[str] | None = None, device: str | None = DEVICE, batch_size: int = BATCH_SIZE, frame_stride: int = FRAME_STRIDE) -> Path:
    """Evaluate completed runs on all dense reference epochs and write paper figures.

    Truth and acquisition metadata come from the public generator's dataset.json.
    The fixed cadence/noise matrix uses the version-1 run folders.
    """
    from spinnme import load
    from spinnme.exports import expand_parameters

    manifest_path = Path(dataset).resolve()
    base = manifest_path.parent
    manifest = _inputs(manifest_path, Path(runs_directory).resolve(), run_ids)
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer.")
    frames = manifest["dense_truth"]["frames"][::frame_stride]
    if not frames:
        raise ValueError("Synthetic evaluation needs a nonempty dense truth sequence.")
    seconds = _time_seconds([frame["time"] for frame in frames])
    if not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0):
        raise ValueError("Dense truth epochs must be finite and strictly increasing.")
    runs = manifest["runs"]
    if run_ids is not None:
        missing = set(run_ids) - {row["run_id"] for row in runs}
        if missing:
            raise ValueError("Unknown synthetic run IDs: " + ", ".join(sorted(missing)))
        runs = [row for row in runs if row["run_id"] in run_ids]
    if not runs or len({row["run_id"] for row in runs}) != len(runs):
        raise ValueError("Select at least one run, with unique run IDs.")
    parameters = expand_parameters(parameters or DEFAULT_PARAMETERS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame_rows, run_rows, records = [], [], []
    evaluation_inputs = []
    for frame in frames:
        item = {"time": frame["time"]}
        for key in ("reference_fits", "truth_npz"):
            path = _resolve(base, frame[key])
            digest = checksum(path)
            if frame.get(key + "_sha256") and frame[key + "_sha256"] != digest:
                raise ValueError(f"Declared synthetic truth checksum differs for {path}.")
            item.update({key: str(path), key + "_sha256": digest})
        evaluation_inputs.append(item)
    for run in runs:
        factors = {key: run.get(key) for key in FACTORS}
        factors["noise_sigma"] = run.get("noise_sigma", run.get("noise_std"))
        factors["noise_level"] = run.get("noise_label", run.get("noise_level"))
        if run.get("noise_std") is not None:
            noise_std = np.asarray(run["noise_std"], dtype=float)
            if noise_std.shape != (4,) or not np.isfinite(noise_std).all() or (noise_std < 0).any():
                raise ValueError("noise_std must contain four finite nonnegative I/Q/U/V values.")
            factors["noise_std"] = json.dumps(noise_std.tolist(), separators=(",", ":"))
        for key in ("cadence_s", "exposure_s", "phase_s"):
            if factors[key] is None or not np.isfinite(factors[key]) or factors[key] < (0 if key == "phase_s" else np.finfo(float).eps):
                raise ValueError(f"Run {run['run_id']} needs a finite valid {key}.")
        if factors["noise_sigma"] is not None:
            factors["noise_sigma"] = json.dumps(factors["noise_sigma"], sort_keys=True)
        identity = {"run_id": run["run_id"], **factors}
        observed_times = run.get("observed_times") or [row["time"] for row in run.get("observations", [])]
        subsets = _subsets(seconds, _time_seconds(observed_times))
        if any("base_grid" in frame for frame in frames):
            subsets["base_grid"] = np.asarray([bool(frame.get("base_grid", False)) for frame in frames])
        snapshot_path = _resolve(base, run["snapshot"])
        by_parameter: dict[str, list[dict]] = defaultdict(list)
        with load(snapshot_path, device=device) as model:
            for index, frame in enumerate(frames):
                truth_path, reference_path = (_resolve(base, frame[key]) for key in ("truth_npz", "reference_fits"))
                truth, truth_metadata = read_npz(truth_path)
                result = model.evaluate(observation=reference_path, time=frame["time"], parameters=parameters,
                                        batch_size=batch_size, instrument_id=run.get("instrument_id"))
                reference_metadata = _checked_metadata(result, truth_metadata, frame["time"], parameters)
                for name in parameters:
                    if name not in truth or name not in result.arrays:
                        raise ValueError(f"Missing {name} in dense truth or prediction at {frame['time']}.")
                    factor = _unit_factor(result.units.get(name), reference_metadata.get("units", {}).get(name), name)
                    predicted = np.asarray(result.arrays[name], dtype=float) * factor
                    mask = _broadcast_mask(result.mask, predicted.shape).copy()
                    if "mask" not in truth:
                        raise ValueError("Dense truth requires its native spatial validity mask.")
                    mask &= _broadcast_mask(truth["mask"], predicted.shape)
                    metric, difference = comparison(predicted, truth[name], mask,
                                                     period=180.0 if name == "azimuth" else None)
                    valid = np.isfinite(difference)
                    row = {**identity, "parameter": name, "unit": reference_metadata["units"][name],
                           "frame_index": index, "time": frame["time"], "time_s": float(seconds[index]-seconds[0]),
                           "sample_class": "observed" if subsets["observed"][index] else
                               "interpolation" if subsets["interpolation"][index] else "extrapolation",
                           **metric, "mean_prediction": float(np.mean(predicted[valid])),
                           "mean_truth": float(np.mean(truth[name][valid]))}
                    frame_rows.append(row)
                    by_parameter[name].append(row)
        for name, rows in by_parameter.items():
            for subset, selected in subsets.items():
                if selected.any():
                    run_rows.append({**identity, "parameter": name, "unit": rows[0]["unit"], "subset": subset,
                                     **_temporal_statistics(rows, seconds, selected)})
        records.append({**identity, "snapshot": str(snapshot_path), "snapshot_sha256": checksum(snapshot_path),
                        "observed_times": observed_times, "evaluation_frame_count": len(frames)})
    summary = _aggregate(run_rows)
    _csv(output / "frame_metrics.csv", frame_rows)
    _csv(output / "run_metrics.csv", run_rows)
    _csv(output / "summary.csv", summary)
    written = _figures(frame_rows, summary, output / "figures")
    write_json(output / "evaluation.json", {
        "schema_version": 1, "created_utc": utc_now(), "dataset": str(manifest_path),
        "dataset_sha256": checksum(manifest_path), "parameters": parameters,
        "device": device, "batch_size": batch_size, "frame_stride": frame_stride, "runs": records, "evaluation_inputs": evaluation_inputs,
        "target": "Same dense instantaneous noiseless latent truth for every cadence/exposure/noise run.",
        "aggregation": "Equal weight per epoch within each run and subset. Pixel/frame counts are descriptive, not independent replicates. The base_grid subset retains uniform latent-grid sampling.",
        "temporal_metrics": "Temporal means use each epoch's common finite spatial/spectral mask; derivatives use actual time separations. Spatial cancellation can reduce temporal-mean errors; full-field RMSE is also reported.",
        "ambiguity": "B, B_los and B_transverse are invariant to the unresolved transverse-field sign. Any explicitly requested Br/Btheta/Bphi are branch dependent; no disambiguation is performed.",
        "files": [{"path": str(path), "sha256": checksum(path)} for path in
                  [output / "frame_metrics.csv", output / "run_metrics.csv", output / "summary.csv", *written]]})
    return output / "summary.csv"


def _figures(frame_rows: list[dict], summary: list[dict], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter

    output.mkdir(parents=True, exist_ok=True)
    written = []
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
             "axes.spines.right": False, "svg.hashsalt": "spinnme-synthetic", "savefig.bbox": "tight"}
    with plt.rc_context(style):
        for parameter in sorted({row["parameter"] for row in frame_rows}):
            selected = [row for row in frame_rows if row["parameter"] == parameter]
            unit = selected[0]["unit"]
            fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.3), sharex=True, layout="constrained")
            by_run = defaultdict(list)
            for row in selected:
                by_run[row["run_id"]].append(row)
            first = next(iter(by_run.values()))
            axes[0].plot([row["time_s"] for row in first], [row["mean_truth"] for row in first],
                         color="black", lw=1.6, label="Instantaneous truth")
            for run_id, rows in by_run.items():
                label = f"{run_id}"
                line = axes[0].plot([row["time_s"] for row in rows], [row["mean_prediction"] for row in rows],
                                    lw=.9, alpha=.75, label=label)[0]
                axes[1].plot([row["time_s"] for row in rows], [row["rmse"] for row in rows],
                             color=line.get_color(), lw=.9, alpha=.75)
                observed = [row for row in rows if row["sample_class"] == "observed"]
                axes[1].scatter([row["time_s"] for row in observed], [row["rmse"] for row in observed],
                                color=line.get_color(), s=10)
            axes[0].set(ylabel=f"Spatial/spectral mean [{unit}]", title=parameter)
            axes[1].set(xlabel="Time from first dense epoch [s]", ylabel=f"Full-field RMSE [{unit}]")
            for axis in axes:
                axis.ticklabel_format(axis="both", style="plain", useOffset=False)
            if len(by_run) <= 12:
                axes[0].legend(frameon=False, fontsize=6, ncol=2)
            written += save_figure(fig, output / f"{parameter}_temporal", ("pdf", "svg", "png"), 300)
            plt.close(fig)
            comparison_rows = [row for row in summary if row["parameter"] == parameter
                               and row["metric"] == "rmse" and row["subset"] == "dense"]
            fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
            groups = defaultdict(list)
            for row in comparison_rows:
                groups[(row["noise_level"], row["noise_sigma"], row["noise_std"], row["exposure_s"], row["phase_s"])].append(row)
            vectors_by_label = defaultdict(set)
            for level, sigma, vector, exposure, phase in groups:
                vectors_by_label[level or sigma].add(vector)
            for (level, sigma, vector, exposure, phase), rows in sorted(groups.items(), key=lambda item: str(item[0])):
                rows.sort(key=lambda row: row["cadence_s"])
                noise_label = str(level or sigma)
                if len(vectors_by_label[level or sigma]) > 1:
                    noise_label += f"; σ IQUV {vector}"
                label = f"{noise_label}; exposure {exposure:g} s; phase {phase:g} s"
                ax.plot([row["cadence_s"] for row in rows], [row["mean"] for row in rows],
                        marker="o", lw=1, label=label)
            ax.set(xlabel="Acquisition cadence [s]", ylabel=f"Dense RMSE [{unit}]", title=parameter)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            low, high = ax.get_ylim()
            minimum_span = .1*max(abs(low), abs(high))
            if high-low < minimum_span:
                # Near-identical runs should not turn numerical roundoff into
                # visually large separations or unreadable eleven-digit ticks.
                midpoint = (low+high)/2
                ax.set_ylim(max(0, midpoint-minimum_span/2), midpoint+minimum_span/2)
            cadences = sorted({row["cadence_s"] for row in comparison_rows})
            if len(cadences) > 1:
                ax.set_xscale("log")
            ax.set_xticks(cadences)
            cadence_formatter = ScalarFormatter(useOffset=False)
            cadence_formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(cadence_formatter)
            ax.xaxis.set_minor_locator(NullLocator())
            if len(groups) <= 12:
                ax.legend(frameon=False, fontsize=6)
            written += save_figure(fig, output / f"{parameter}_cadence_noise", ("pdf", "svg", "png"), 300)
            plt.close(fig)
    return written


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Public synthetic dataset.json')
    parser.add_argument('--runs', required=True, type=Path, help='Parent directory of the version-1 synthetic run folders')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default=DEVICE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--frame-stride', type=int, default=FRAME_STRIDE)
    args = parser.parse_args(argv)
    try:
        print(evaluate_series(args.data, args.runs, args.output, parameters=DEFAULT_PARAMETERS,
                              device=args.device,
                              batch_size=args.batch_size,
                              frame_stride=args.frame_stride))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f'synthetic evaluation: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
