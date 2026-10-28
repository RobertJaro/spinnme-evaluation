#!/usr/bin/env bash
# Download native 90-second and 720-second HMI Stokes observations.
set -euo pipefail

# Fixed HPC download paths.
DATA90="/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_90s"
DATA720="/glade/work/rjarolim/data/hmi_stokes/20240323_0100_0300_720s"
JSOC_EMAIL="robert.jarolim@uni-graz.at"

OPTIONS=(--start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00
         --time-scale tai --email "$JSOC_EMAIL")

# Native 90-second Stokes observations.
spinnme download hmi --product S_90s --cadence 90s "${OPTIONS[@]}" --output "$DATA90"

# Native 720-second Stokes observations.
spinnme download hmi --product S_720s --cadence 720s "${OPTIONS[@]}" --output "$DATA720"
