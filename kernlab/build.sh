#!/bin/bash
# usage: build.sh src.comp out.spv [-Dflags]
C=/home/dylan/AI/engine-lab/tools/sdk/1.4.363.0/x86_64/bin/glslc
src=$1; out=$2; shift 2
$C --target-env=vulkan1.3 -fshader-stage=compute -O "$@" $src -o $out
