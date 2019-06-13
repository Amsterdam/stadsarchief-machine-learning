#!/usr/bin/env bash

echo "----- CUDA ------"
nvcc --version

echo "----- cuDNN ------"
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
