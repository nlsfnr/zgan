#!/bin/bash

rsync \
    --links \
    --recursive \
    --verbose \
    nf711@viking.york.ac.uk:~/scratch/zgan/viking/jobs/ \
    ./viking-results/jobs/

rsync \
    --links \
    --recursive \
    --verbose \
    nf711@viking.york.ac.uk:~/scratch/zgan/zoo/ \
    ./viking-results/zoo/
