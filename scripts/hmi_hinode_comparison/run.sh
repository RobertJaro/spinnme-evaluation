#!/bin/bash -l

#PBS -N SPINNME
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

cd /glade/work/rjarolim/spinnme-evaluation

spinnme run --config configs/hmi_hinode_comparison/hinode.yaml
