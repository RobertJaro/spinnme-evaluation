#!/usr/bin/env bash
set -euo pipefail
cd /glade/work/rjarolim/spinnme-evaluation

# 5-second observations.
spinnme run --config configs/synthetic_validation/cadence5s-noiseless.yaml
spinnme run --config configs/synthetic_validation/cadence5s-optimistic.yaml
spinnme run --config configs/synthetic_validation/cadence5s-nominal.yaml
spinnme run --config configs/synthetic_validation/cadence5s-challenging.yaml

# 30-second observations.
spinnme run --config configs/synthetic_validation/cadence30s-noiseless.yaml
spinnme run --config configs/synthetic_validation/cadence30s-optimistic.yaml
spinnme run --config configs/synthetic_validation/cadence30s-nominal.yaml
spinnme run --config configs/synthetic_validation/cadence30s-challenging.yaml

# 90-second observations.
spinnme run --config configs/synthetic_validation/cadence90s-noiseless.yaml
spinnme run --config configs/synthetic_validation/cadence90s-optimistic.yaml
spinnme run --config configs/synthetic_validation/cadence90s-nominal.yaml
spinnme run --config configs/synthetic_validation/cadence90s-challenging.yaml

# 180-second observations.
spinnme run --config configs/synthetic_validation/cadence180s-noiseless.yaml
spinnme run --config configs/synthetic_validation/cadence180s-optimistic.yaml
spinnme run --config configs/synthetic_validation/cadence180s-nominal.yaml
spinnme run --config configs/synthetic_validation/cadence180s-challenging.yaml

# 720-second observations.
spinnme run --config configs/synthetic_validation/cadence720s-noiseless.yaml
spinnme run --config configs/synthetic_validation/cadence720s-optimistic.yaml
spinnme run --config configs/synthetic_validation/cadence720s-nominal.yaml
spinnme run --config configs/synthetic_validation/cadence720s-challenging.yaml

