# SPINNME evaluation

Reusable array comparisons and plotting live in the core framework:
`spinnme.evaluation.comparison` and `spinnme.plotting` (`plot_comparison`,
`plot_maps`, `plot_scatter`, `save_figure`). This project contains dataset pairing,
reference conventions, unit fallbacks, study choices, paths and figure assembly.
The HMI reference and cadence scripts call the shared plotting functions;
`evaluation/_common.py` re-exports core metrics/export helpers for the other studies.

```bash
python -m pip install -r /glade/work/rjarolim/spinnme-evaluation/requirements.txt
wandb login

WORKFLOW=hmi_cadence  # or synthetic_validation / phi / phi_fdt
bash /glade/work/rjarolim/spinnme-evaluation/scripts/$WORKFLOW/download.sh
bash /glade/work/rjarolim/spinnme-evaluation/scripts/$WORKFLOW/run.sh
bash /glade/work/rjarolim/spinnme-evaluation/scripts/$WORKFLOW/evaluation.sh
```

Run commands select one YAML each, with run outputs (`path`) under `/glade/work/rjarolim/spinn_me` and work caches (`work_path`) under `/glade/derecho/scratch/rjarolim/<run-name>`. Config paths are relative to the checkout root; scripts enter `/glade/work/rjarolim/spinnme-evaluation` first. Data paths remain under `/glade/work/rjarolim/data`. All runs use version 1 (`_v01`).

HMI download scripts use `robert.jarolim@uni-graz.at`. The cadence download script downloads both data series first, then their transmission profiles in a separate section.

Training scripts contain explicit commands, with no automatic reuse or resume. Run only the desired commands; add `--resume` explicitly to continue an existing checkpoint. Devices are selected automatically. W&B project is `spinnme`; set `logging.mode: disabled` in the run YAML to skip W&B. Training validation images are enabled in every run YAML and logged to W&B every `training.val_check_interval` batches (HMI: 2000; PHI/synthetic: 1000). Paper-plot settings live in `evaluation/*.py`.

The HMI cadence workflow compares only the native 90s and 720s products. HMI–Hinode reuses the existing HMI snapshot for its comparison. Its separate `run.sh` trains a joint Hinode Fe I 6301/6302 inversion from calibrated Stokes spectra. Edit analysis constants directly in the corresponding `evaluation/*.py` script. YAML files are only for inversion runs and synthetic data generation. Synthetic generation and all training/downloads use the base SPINNME package; Python files here only produce analysis and plots.

The Hinode reference available in this interval is the [DARTS MERLIN raster
20240323_020035](https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2hao/2024/03/23/SP3D/20240323_020035/20240323_020035.fits),
with 950 slit epochs from 02:00:35.666 to 03:05:13.881 UTC. Evaluation uses
Hinode's native X/Y coordinates and keeps slit times inside the requested
interval. The alignment shift is applied to Hinode sky coordinates first;
physical xyzt and viewing matrices are then derived from this corrected frame.
SPINNME HMI is evaluated at these corrected positions. SPINNME Hinode is loaded
at its original unshifted coordinates and displayed at the shifted positions.
Both use the original per-slit times.
The nearest HMI B_720s field/inclination maps (within 360 s; earlier wins ties)
are bilinearly reprojected onto this same grid; azimuth uses nearest-neighbor
sampling. Reference epochs are not substituted for model evaluation times.
HMI records just outside the window remain available to bracket edge pixels.

There is no automatic registration, coordinate fitting or PSF smoothing. Hinode values
remain on their native grid. Existing HMI Stokes and confidence files are not
required. By default the comparison uses finite physical MERLIN values;
`--hinode-fit-status` optionally restricts the fits.
This raster has statuses 1, 4 and 5, with no status-3 pixels. See the
[HAO status definitions](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description).
Native instrument resolutions differ; these are direct map comparisons.

`--shift-x` and `--shift-y` add arcseconds to the Hinode X/Y positions. The
shell script uses `--shift-x 25 --shift-y 30` for the approximate alignment.
HMI reference and both SPINNME maps share the shifted Hinode display grid.
MERLIN retains its native pixels, translated into that same sky frame.
`--hpc-range` crops the common grid and sets all four plot limits. Without an
explicit crop, the plot includes the grids with the default solar-X upper limit
of -110 arcsec. Python defaults are zero shift.

```bash
bash scripts/hmi_hinode_comparison/download.sh  # MERLIN reference + calibrated Stokes
bash scripts/hmi_hinode_comparison/run.sh       # independent Hinode inversion
bash scripts/hmi_hinode_comparison/evaluation.sh
# Optional spatial crop, in arcseconds:
bash scripts/hmi_hinode_comparison/evaluation.sh --hpc-range -237 -157 -152 -72
```

The Stokes download selects archive filename times in 00:59:23–02:59:23 UTC (the same
01:00–03:00 TAI comparison window). It downloads calibrated Level-1 I/Q/U/V
from DARTS `level1hao` into the configured download directory, with a checksum manifest. Download
selection uses only the hard-coded times, without a reference file. The Hinode
dataset assembles the downloaded slit files in memory during inversion. The YAML
points directly to their folder; no prepared FITS files or copied slit data are
created. Both Fe I lines retain their native sampling, slit positions and UTC
times, including gaps in slit indices. Geometry comes from the calibrated slit
headers. The observer is approximated by Earth at each slit’s acquisition time,
with its own roll and any supplied observer-velocity components. Native slit geometry and timing are saved in the
inference snapshot. The +25/+30 arcsec translation remains specific to the
MERLIN comparison; it is not applied to the native Stokes data.

The run uses `configs/hmi_hinode_comparison/hinode.yaml`, saves to
`/glade/work/rjarolim/spinn_me/hinode_sp_20240323_020035_v03`, and uses the bundled
joint-line settings (24.8 mÅ FWHM Gaussian spectral convolution with nine
quadrature nodes, ±0.30 Å windows, 100,000 steps, validation of both lines every
1,000 batches). The Stokes profile comparison shows both lines in separate
wavelength panels.
The run and work paths use the `_v03` suffix. Rebuild an existing preparation
cache once with `--reload` to apply the updated fitting windows and both validation lines. Slit coordinates and times are retained without interpolation.
These spectral settings match MERLIN’s documented response and windows; the
run does not implement MERLIN’s empirical scattered-light/filling-factor model. `run.sh` contains only the inversion CLI command. The existing
comparison includes this snapshot alongside the HMI snapshot and uses MERLIN as its reference.
The comparison loads the Hinode `inversion.pme` supplied with `--hinode-result`.
`--hinode-observation train_01_000` selects its saved Fe I 6302 grid; without an
ID, the first training Fe I 6302 observation is used. An existing FITS-directory
argument follows its provenance back to the source snapshot.

SPINNME Hinode retains its native unshifted prediction. SPINNME HMI uses
Cartesian `(x,y,z,t)` derived from the shifted Hinode frame and its saved
per-slit observer metadata. The alignment changes Hinode display positions,
not the coordinates used to evaluate its own model. HMI sampling
converts only spatial/time normalization and time origin into the HMI model's
units. It runs in one batched pass with one progress bar. Native slit times and
pixel ordering are retained; the snapshot files themselves are not modified.
Azimuths are displayed in the matched HMI image basis. No intermediate FITS
export is needed.

The comparison follows five stages in `evaluation/hmi_hinode_comparison.py`:
load, align Hinode, derive common geometry, sample/reproject, and plot. Every map
carries explicit X/Y coordinates and a validity mask. Maps and distributions use
the same display bounds. Cumulative LOS uses absolute values, and both field
distributions use symlog ticks at 0, 1, 10, 100, and so on.

The hard-coded paths follow `scripts/hmi_cadence`, using the existing HMI90 v01
snapshot without physics and HMI B_720s references. The interval is
`--start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00 --time-scale tai`,
matching the inversion archive request (00:59:23–02:59:23 UTC).
Pass `--time-scale utc` to interpret those bounds as UTC instead. Additional CLI
arguments are forwarded. The full raster within the time window is used by
default; `--hpc-range` selects a crop.

Results include one combined 3×4 comparison plate (`comparison.png`/`.pdf`):
BLOS, Btrv and azimuth modulo 180° in rows; HMI reference, SPINNME HMI, Hinode MERLIN and SPINNME Hinode in columns.
All panels have a maximum Solar X of −110 arcsec. This is a display crop; no
slit columns are removed from the input data.
The plate matches the HMI cadence comparison’s compact panel spacing, fonts,
sparse coordinate ticks and narrow shared colorbars. BLOS uses symlog with a
1 G linear threshold, Btrv uses log with a 1 G minimum, and azimuth uses a cyclic
0–180° scale with CMasher’s white-free `infinity` colormap. `--bmax` defaults to 3000 G and `--dpi` to 400, as in the cadence
comparison. Matching HMI `azimuth` files are required;
native azimuth pixels are sampled without interpolating across the 180° seam.
Hinode azimuth is rotated by +90° and wrapped to 0–180° to match the
comparison convention; the correction applies to the plotted values.
Azimuth maps show all finite values at full opacity. A separate column after
the colorbars shows cumulative distributions on
each source's finite native pixels inside the same displayed sky bounds.
The Hinode FITS curve is not restricted to MERLIN pixel matches; the curves
can contain different pixel counts because the grids differ. The vertical axis gives the
percentage of pixels at or below each value; distinct colors and line styles
identify the sources without overlapping fills or bin choices. Both magnetic
axes use symlog with a 1 G linear threshold, retaining zero values and tails
outside the map color limits. The azimuth distribution retains field-strength
weights of zero at ≤10 G, increasing linearly to one at 100 G; its cumulative
curve starts at the chosen 0° branch.
Constrained layout reserves space for axis labels and the dedicated colorbars.
The script saves only the combined plot as `comparison.png` and
`comparison.pdf`. Rerunning replaces these plots in the output directory.


`scripts/hmi_cadence/evaluation.sh` has separate sections for the original
single-savestate HMI720 comparison and the four-column cadence publication plates.
Running the shell script produces both. To run only the single-savestate
comparison, call `evaluation/hmi_reference.py` directly:

```bash
python evaluation/hmi_reference.py \
  --input /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_physics_v02/inversion.pme \
  --reference /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s \
  --time 2024-03-23T02:00:00 \
  --output /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_physics_v02/evaluation_720s \
  --hpc-range -200 200 -200 200
```

This utility adapts the original `pinn-me` archive's `load_ref_map.py` and
`load_ref_comparison.py` to the public SPINNME snapshot API.
The following single-savestate output and disambiguation options apply to
`hmi_reference.py`; the four-column cadence evaluation is described below.

Outputs are 400 dpi PNG and PDF figures: `inversion_parameters` (field strength,
inclination and azimuth), `magnetic_vector` (Br, Btheta, Bphi), `los_velocity`,
`velocity_vector` (model vr, vtheta, vphi) and `scatter`. Comparison maps show HMI,
SPINNME and their difference with shared scales and solar coordinates. NPZ maps,
per-parameter masks, `metrics.csv` and `comparison.json` retain values and choices.

Append `--hpc-range XMIN XMAX YMIN YMAX` in arcsec for an optional subframe.
The longest output dimension defaults to 1024; `--resolution 0` keeps native
resolution. Sampling uses nearest native pixels, including for angular maps.
`--time` selects the nearest actual reference epoch; omitting it selects the
middle reference record. The effective T_OBS epoch (TAI in JSOC) is used for model evaluation and coordinate geometry; DATE-OBS is the exposure start.

Azimuth residuals wrap modulo 180 degrees; inclination remains in [0,180].
Brtp applies the paired HMI `disambig` bitmap before the vector conversion:
bit 0 for `conf_disambig` 90/60 (minimum-energy/smoothed solution), bit 2 for
confidence 50 (weak-field radial-acute solution). Unresolved confidence 0 pixels
are excluded from signed-component comparisons. The SPINNME branch is retained;
the reference is never flipped toward the model. See the
[HMI pipeline definition, Appendix A.4](https://arxiv.org/html/1404.1881#A1.SS4).

Both 90 s and 720 s full-disk `azimuth` require this correction. Already-disambiguated
SHARP inputs are recognized; for other explicitly preprocessed azimuth maps use
`--azimuth-disambiguated` to avoid applying the bitmap twice. Missing bitmap or
confidence maps warn and skip Brtp figures/metrics while preserving the remaining
comparisons. The HMI720 reference download includes the required maps:

```bash
bash scripts/hmi_cadence/download_reference.sh
# Then rerun the hmi_reference.py command above.
```

Disambiguation status, chosen bits and flipped/valid pixel counts are recorded in
`comparison.json`. Azimuth plots and errors remain modulo 180 degrees.
Vlos is positive redshift: the script converts
the FITS velocity units and subtracts observer motion, retaining solar rotation
to match SPINNME `v_los`. No velocity offset is fitted. Vr/Vtheta/Vphi are model
velocities in the Carrington frame; HMI provides no vector-velocity reference.
The pipeline is an operational reference, not ground truth; no PSF matching is
performed. The separate `evaluation/hmi_cadence.py` retains the cadence study.

In its second section, `evaluation.sh` calls
`evaluation/hmi_cadence_comparison.py` with four input paths: two SPINNME `.pme`
files (`--input90`, `--input720`) and two HMI reference paths (`--reference90`,
`--reference720`). Each reference path can select a `*.field.fits` file with
paired inclination/azimuth files beside it, or a directory of reference files. Run:

```bash
bash scripts/hmi_cadence/evaluation.sh
```

This writes 400 dpi `cadence_full_disk.png` and `cadence_zoom.png` to
`/glade/work/rjarolim/spinn_me/hmi_cadence_publication`. Columns are HMI reference
90 s, HMI reference 720 s, SPINNME 90 s, SPINNME 720 s. Rows show field strength
(viridis, log 1–3000 G), inclination (PiYG, 0–180 degrees), and azimuth
(twilight, modulo 180 degrees). Each row shares one colorbar; coordinate labels
appear only along the left and bottom edges. Only the full-disk plate has column
titles. Red boxes mark the zoom in every full-disk panel.

Override the initial zoom with `bash scripts/hmi_cadence/evaluation.sh --zoom
XMIN XMAX YMIN YMAX`, in HPC arcsec; the default is `0 100 0 50`. The zoom is
evaluated separately at native reference resolution, rather than cropped from
the reduced full-disk arrays. `--resolution` controls the full-disk maximum
dimension (1024 by default, 0 for native); `--zoom-resolution` optionally limits
the zoom dimension. `--bmax` changes the shared upper field limit in both plates.

The script selects the actual HMI720 epoch nearest 02:00 UTC, its nearest HMI90
record within half a 90 s cadence, and evaluates both snapshots at that HMI720
epoch on the same observing grid. No JSON sidecar is written. Three paired magnetic FITS
segments are required per reference (`field`, `inclination`, `azimuth`);
nonzero QUALITY is rejected. Eight NPZ products retain the plotted maps/WCS.
The shell script currently selects the 90 s physics v02 snapshot and the 720 s
v01 snapshot; set `--input90` / `--input720` to compare other runs.

HMI reference headers are handled permissively: missing `BUNIT` uses the standard
segment units (field: G; inclination/azimuth: degrees; vlos_mag: cm/s). Explicit
units are converted when supplied. Nonzero QUALITY and segment time/observer
metadata differences warn rather than abort. Each segment is sampled using its
own WCS, so different image dimensions are supported. Unavailable observer
velocity components default to zero. Warnings and unit assumptions are recorded
in `comparison.json`; missing required images, unusable WCS, incompatible explicit
units and an empty comparison still fail.

The `phi` workflow downloads the March 23–24 HRT Stokes series, trains with artifact correction, and exports/plots maps and Stokes residuals in `evaluation.sh`. It assumes the released files already contain correct WCS and calibrated wavelength metadata.

## PHI-FDT download, inversion and pipeline comparison

Use the current SPINNME checkout containing the Airy response implementation:

```bash
python -m pip install -e '/glade/work/rjarolim/spinnme[download,wandb]'
bash scripts/phi_fdt/download.sh
bash scripts/phi_fdt/run.sh
bash scripts/phi_fdt/evaluation.sh
```

The default interval is **23 March 2024, 00:00 UTC to 24 March 00:00 UTC**. A full day accommodates sparse FDT synoptic observations and predates the routine onboard averaging introduced in 2025. Archive coverage of this exact interval is checked when the download runs. Edit the interval and the paths in the scripts, `configs/phi/fdt.yaml` and `evaluation/phi_fdt.py` together for another study. This workflow creates cluster-ready commands; the checked-in tests use synthetic fixtures, not downloaded solar observations.

Each product (`stokes`, `bmag`, `binc`, `bazi`, `blos`, `vlos`) has its own download directory and SPINNME checksum manifest. The preflight requires **archive VERSION 3**, pairs by `PHIDATID` and `DATE-OBS`, checks integration metadata, observer geometry, native WCS and units, and rejects duplicates/missing pairs. It writes `pairs.json` in the collection root. The run suffix `_v01` denotes our experiment and is independent of archive version 3. A different archive version requires an explicitly selected clean collection and a matching `--version` setting; no old/new versions are silently merged. Reusing an existing matching download manifest does not query the archive for newer processing; use a new collection for an archive refresh.

The inversion uses the new nominal Airy response (106 mÅ FWHM, 3.01 Å FSR, 161 quadrature samples), fixed filling factor and no learned artifact, limb or velocity corrections. It retains native pixels with `mu >= 0.2`. The initial budget is 10,000 steps, with 512 profiles per batch; increase the budget and check passband quadrature convergence before scientific interpretation. W&B validation plots are enabled. The baseline keeps the same calibrated six wavelength positions in the data and forward model.

The evaluation exports native NPZ maps and spectra, then writes under `/glade/work/rjarolim/spinn_me/evaluation/phi_fdt`:

- `metrics.csv` and `comparison.json`: bias, MAE, RMSE, median/p95 absolute error, paired counts and Pearson correlation for field strength, LOS field, transverse field, inclination, wrapped azimuth and relative velocity. Angular residuals use a 180° period for azimuth; angular Pearson correlation is omitted.
- `stokes_metrics.csv` and per-acquisition `stokes.png/pdf`: normalized I/Q/U/V residuals, mean profiles and wavelength-resolved RMS errors.
- Per-parameter PNG/PDF figures: pipeline map, SPINNME map, difference and density scatter plot, with shared scales and native pixel coordinates.
- Per-acquisition `masks.npz`: the common finite mask and parameter-specific masks. Inclination/azimuth metrics require total/linear polarization above five times a robust continuum-noise estimate; the estimates are saved in JSON.
- Input, export and snapshot checksums, acquisition/release metadata and averaging/quality cards for provenance. Inspect the official quality inventory and mask remaining known defects before interpreting the metrics.

The pipeline maps are an **operational reference, not ground truth**. Magnetic LOS is positive toward the observer; azimuth is counterclockwise from detector +y, compared modulo 180°. No signed transverse/spherical-field comparison or disambiguation is attempted. `B_transverse` is derived from pipeline `bmag` and `binc` on their native grid. Units are read from FITS `BUNIT`, including conversion of km/s to m/s; missing/incompatible units fail explicitly. Maps are never automatically rotated, resampled or registered, and no PSF matching is performed.

Velocity comparison uses **`forward_vdop`**, which includes the observer/instrument terms used to fit the supplied spectra. It removes one common-mask median SPINNME-minus-pipeline offset per acquisition and saves that offset as `removed_velocity_offset_m_s`; the reported relative error cannot validate an absolute velocity zero point or repair spatial calibration errors. Observer-corrected `Vdop` is not directly compared to raw pipeline `vlos`. See the [official FDT release notes](https://www.mps.mpg.de/solar-physics/solar-orbiter-phi/data-releases) for partial spacecraft-velocity compensation, residual cavity/ghost/fringe effects and temporal averaging.

Comparison output must be new or empty. To rerun analysis, choose a fresh directory:

```bash
python evaluation/phi_fdt.py --check-only
python evaluation/phi_fdt.py --output /glade/work/rjarolim/spinn_me/evaluation/phi_fdt_repeat
python -m pytest tests/test_phi_fdt.py -q
```

## HMI90 inversion with physics

```bash
spinnme run --config configs/hmi_cadence/hmi90_physics.yaml
```

This uses the baseline data, model architecture, training, Stokes normalization
and instrument settings from `hmi90.yaml`, with separate output/cache paths and
W&B run name. Stokes synthesis and physics use the same magnetic field, with
no additional transverse rotation or learned disambiguation mask.
Physics samples 4,096 points per step between **0.99 and 1.01 solar radii**, over
the training observations' angular footprint and time range. Characteristic
normalization uses **10 Mm** and **1000 G**, independent of dataset units.
Divergence and potential-field (curl-free) weights each decay exponentially
from **1** to **1e-4** over 200,000 steps, without warmup. The baseline
direct-field model is retained,
so both constraints are soft penalties; inspect the trained physics losses
alongside the Stokes residuals. No upper or side boundary conditions are imposed.

### Magnetic-field comparison video

`evaluation/blos_video.py` compares a 90 s-trained snapshot sampled every 10 s
with the closest HMI `B_720s` inversion for BLOS, transverse field and azimuth.
The `field`, `inclination` and `azimuth` FITS segments are needed. For the server paths, run:

```bash
# Video section of the cadence evaluation script:
bash scripts/hmi_cadence/evaluation.sh
```

Edit the hard-coded `--center-carrington` in the shell script to select a region
(currently 210°, −10°). Shell launchers do not forward
external arguments. The launcher uses
2024-03-23 01:00–03:00 UTC and a 512×512 pixel patch. Run directly for
other inputs or time ranges:

```bash
python evaluation/blos_video.py \
  --input90 /path/to/inversion.pme \
  --reference720 /path/to/hmi_reference_720s \
  --center-carrington 150 -10 \
  --start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00 \
  --output /path/to/blos_comparison_frames
```

The surface grid is centered at `--center-carrington` with east/north tangent
axes and fixed Carrington coordinates. No anchor epoch or anchor reference is
selected. `--pixel-scale` sets the disk-center-equivalent arcsec/pixel, converted
to a fixed surface step using median HMI geometry (native scale by default).
SPINNME is evaluated at **every video frame time**, with observer longitude,
latitude, distance and detector roll interpolated between actual HMI epochs;
angles wrap continuously. Geometry matches HMI at the reference timestamps.
Outside the reference span, endpoint observer geometry is held. Visibility and
LOS/transverse projections are recomputed per frame, so choosing a new nearest
HMI reference does not reset the SPINNME viewing geometry. The model's spatial
footprint stays fixed, and only its time and viewing projection change.
HMI remains sampled at its actual observation time, with its timestamp and time
offset shown. Azimuth is rotated into the current frame's detector basis.
HMI BLOS is formed as `B*cos(inclination)` on the native grid
before bilinear interpolation. Earlier reference epochs win exact midpoint ties.
Reference frames are held without temporal interpolation; their timestamps and
signed time offsets are displayed. Off-disk/invalid pixels are masked, and gaps
beyond `--max-reference-age` (default 360 s) are labeled while retaining the
closest reference. Sampling at 10 s does not imply 10 s observational resolution.

Each frame has three rows (BLOS, transverse field, azimuth) and three columns
(HMI, SPINNME, signed difference). BLOS uses symlog with a linear range of ±1 G;
transverse field uses a log scale starting at 1 G. `--vmax` fixes their upper
bound for the whole video. Azimuth uses a cyclic 0–180° scale. Magnetic
residuals use symlog with a 1 G linear threshold and `--difference-max` limits;
azimuth residuals wrap to [−90°, 90°). HMI transverse strength is calculated
before interpolation; azimuth is interpolated through sin(2χ)/cos(2χ) to avoid
the 0/180° seam. Azimuth uses the current frame's detector basis for both
sources; no 180° disambiguation is needed. It loads the snapshot once,
reads only cropped FITS regions, caches HMI maps until the reference changes,
and recomputes model geometry at each frame time. It exports only sequential
JPEG frames (`frame_000000.jpg`, `frame_000001.jpg`,
…). No full-disk reprojection or Stokes synthesis is performed. `--output`
selects a new or empty frame directory. No video encoding or sidecar files are
written, and FFmpeg is not required. `--device cpu` supports CPU evaluation.

HMI comparisons use the saved model and geometry dates. There is no coordinate
version check or warning for HMI timing, and no automatic model-time shift.
Exports retain both dates; the HMI azimuth basis is unchanged. Legacy Hinode
snapshots retain their saved polarization interpretation with a warning.
Both MERLIN and SPINNME Hinode azimuths are expressed in each matched
HMI image basis, including its roll. The MERLIN conversion is
`(azimuth + 90 + degrees(HMI pAng)) % 180`; the audited calibrated profiles do
not support negating MERLIN azimuth despite the CSAC clockwise description.
HMI predictions use the exact xyzt and projection derived from the shifted Hinode frame.
Matched HMI reference records affect the displayed azimuth basis only, not the
sampling coordinates or model times.
