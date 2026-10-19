#!/bin/bash
cd ~/AI/engine-lab/kernlab
for m in 0 1 2; do echo BARMODE=$m; BARMODE=$m timeout 60 ./barlab 240 200 | head -1; done
