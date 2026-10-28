#!/usr/bin/env bash
set -euo pipefail
cd /glade/u/home/rjarolim/projects/spinnme-evaluation

# Shift the Hinode frame first, then derive common physical xyzt and projection.
# Reproject/sample HMI there; load Hinode at its unshifted xyzt. Keep slit times.
python evaluation/hmi_hinode_comparison.py \
  --input /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_720s_v01/inversion_converted.pme \
  --hinode-result /glade/work/rjarolim/spinn_me/hinode_sp_20240323_020035_v01/inversion.pme \
  --hinode-observation train_01_000 \
  --reference /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s \
  --hinode /glade/work/rjarolim/data/hinode_merlin/20240323_020035/20240323_020035.fits \
  --shift-x 25 --shift-y 30 \
  --start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00 --time-scale tai \
  --output /glade/work/rjarolim/spinn_me/evaluation/hmi_hinode_comparison
