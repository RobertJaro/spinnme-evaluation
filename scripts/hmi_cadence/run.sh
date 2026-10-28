#!/usr/bin/env bash
set -euo pipefail
cd /glade/work/rjarolim/spinnme-evaluation

# Native 90-second observations.
spinnme run --config configs/hmi_cadence/hmi90.yaml

spinnme run --config configs/hmi_cadence/hmi90_physics.yaml

# Native 720-second observations.
spinnme run --config configs/hmi_cadence/hmi720.yaml
