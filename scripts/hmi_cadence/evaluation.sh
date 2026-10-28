#!/usr/bin/env bash
set -euo pipefail
cd /glade/u/home/rjarolim/projects/spinnme-evaluation

# --- Single-savestate comparison against the HMI720 pipeline ---
python evaluation/hmi_reference.py \
  --input /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_physics_v01/inversion.pme \
  --reference /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s \
  --time 2024-03-23T02:00:00 \
  --output /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_physics_v01/evaluation_720s \
  --hpc-range -200 200 -200 200

# Optional subframe: --hpc-range XMIN XMAX YMIN YMAX (arcsec).
# For native resolution, use --resolution 0; default maximum dimension is 1024.

# --- Four-column cadence comparison: full disk and native-resolution HPC zoom ---
python evaluation/hmi_cadence_comparison.py \
  --input90 /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_v03/inversion.pme \
  --input720 /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_720s_v01/inversion.pme \
  --reference90 /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_90s \
  --reference720 /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s \
  --output /glade/work/rjarolim/spinn_me/evaluation/hmi_cadence_comparison3 \
  --time 2024-03-23T02:00:00 \
  --zoom -400 0 -200 50

# --- Magnetic video: SPINNME90 and smooth observer geometry at each 10 s frame ---
# Fixed Carrington surface patch (degrees); no anchor epoch.
# Edit --center-carrington here to select another region.
python evaluation/blos_video.py \
  --input90 /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_v01/inversion.pme \
  --reference720 /glade/work/rjarolim/data/hmi_reference/20240323_0100_0300_720s \
  --output /glade/work/rjarolim/spinn_me/hmi_fd_20240323_2h_90s_v01/video_comparison \
  --center-carrington 210 -10 \
  --start 2024-03-23T01:00:00 \
  --end 2024-03-23T03:00:00 \
  --shape 512 512 \
  --cadence 10
