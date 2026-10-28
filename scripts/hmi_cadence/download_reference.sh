#!/usr/bin/env bash
set -euo pipefail
cd /glade/u/home/rjarolim/projects/spinnme-evaluation

# Segments required by the two sections in evaluation.sh.
# QUALITY metadata come from FITS headers; no info_map/confid_map is needed.
# Native 90-second reference: field-strength and angle publication panels only.
spinnme download hmi --product B_90s --cadence 90s \
  --start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00 --time-scale tai \
  --segments field inclination azimuth vlos_mag \
  --output /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_90s \
  --email robert.jarolim@uni-graz.at

# Native 720-second reference: publication panels plus the original comparison's
# LOS velocity and signed Brtp maps (disambig/conf_disambig).
spinnme download hmi --product B_720s --cadence 720s \
  --start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00 --time-scale tai \
  --segments field inclination azimuth vlos_mag disambig conf_disambig \
  --output /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s \
  --email robert.jarolim@uni-graz.at
