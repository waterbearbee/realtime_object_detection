#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building nms op..."
cd mmdet/ops/nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building sigmoid focal loss op..."
cd ../sigmoid_focal_loss
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
