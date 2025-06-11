#!/bin/bash
# CPU-only environment variables for Query Cache Service
# Source this file before running the service: source cpu_environment.sh

# Force CPU-only mode for all ML libraries
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM=false
export SPACY_DISABLE_CUDA=1

# Additional settings for stability
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "CPU-only environment variables set:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  TORCH_USE_CUDA_DSA=$TORCH_USE_CUDA_DSA"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "  SPACY_DISABLE_CUDA=$SPACY_DISABLE_CUDA"
