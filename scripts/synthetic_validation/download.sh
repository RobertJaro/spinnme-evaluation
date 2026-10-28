#!/usr/bin/env bash
set -euo pipefail
cd /glade/work/rjarolim/spinnme-evaluation

# Generate dense truth and every cadence/noise stream in a fresh data directory.
spinnme synthetic --config configs/synthetic_validation/dataset.yaml \
  --output /glade/work/rjarolim/data/synthetic_validation/high_frequency
