#!/usr/bin/env bash
set -euo pipefail
cd /glade/work/rjarolim/spinnme-evaluation

RUN="/glade/work/rjarolim/spinn_me/phi/hrt_20240323_v01"
OUTPUT="/glade/work/rjarolim/spinn_me/evaluation/phi"

# Export native maps and spectra using the base package.
spinnme export "$RUN/inversion.pme" --all-observations \
  --parameters Brtp Blos Vdop IQUV Artifact --format npz \
  --output "$RUN/products" --overwrite

# Plot maps, artifact coefficients and Stokes residuals.
python evaluation/phi.py --products "$RUN/products" --output "$OUTPUT"
