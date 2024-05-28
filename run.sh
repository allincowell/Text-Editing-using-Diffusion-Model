#!/bin/bash

# Assign input arguments to variables
# IMAGE_PATH="./segmentation/photos/woman_snow/000000000785.jpg"
IMAGE_PATH="./inputs/boy_dog.jpeg"
OUTPUT_DIR="./inputs/boy_dog"
# SEGMENT_PROMPT="dog"
# REPLACE_PROMPT="cat"

SEGMENT_PROMPT="boy"
REPLACE_PROMPT="girl"

# Run the Python script with the specified arguments
# python ./segmentation/segment.py -i "$IMAGE_PATH" -p "$SEGMENT_PROMPT" -o "$OUTPUT_DIR"
# rm -rf "$OUTPUT_DIR/mask.png"


python main.py -p "$REPLACE_PROMPT" -i "$IMAGE_PATH" --mask "$OUTPUT_DIR/best_segment_boy.png" --output_path "$OUTPUT_DIR/output_boy"