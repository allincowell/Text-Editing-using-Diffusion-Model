#!/bin/bash

# Assign input arguments to variables
# IMAGE_PATH="./segmentation/photos/woman_snow/000000000785.jpg"
IMAGE_PATH="./inputs/000000046804.jpg"
OUTPUT_DIR="./inputs/000000046804"
SEGMENT_PROMPT="sheep"
REPLACE_PROMPT="black dog"
GPU=1

# Run the Python script with the specified arguments
CUDA_DEVICES=$GPU python ./segmentation/segmentation_nvidia.py -i "$IMAGE_PATH" -p "$SEGMENT_PROMPT" -o "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR/mask.png"


CUDA_DEVICES=$GPU python main.py -p "$REPLACE_PROMPT" -i "$IMAGE_PATH" --mask "$OUTPUT_DIR/best_segment.png" --output_path "$OUTPUT_DIR/output"