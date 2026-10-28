#!/usr/bin/env bash
set -euo pipefail
cd /glade/work/rjarolim/spinnme-evaluation

python evaluation/synthetic_validation.py \
  --data /glade/work/rjarolim/data/synthetic_validation/high_frequency/dataset.json \
  --runs /glade/work/rjarolim/spinn_me/synthetic_validation \
  --output /glade/work/rjarolim/spinn_me/evaluation/synthetic_validation
