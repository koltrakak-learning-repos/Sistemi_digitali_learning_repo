#!/bin/bash

PROGRAM="./image_proc"

# Define the input image
INPUT_IMAGE="./flower.jpg"

# Function to run the program and display the command
run_example() {
    echo "Running: $PROGRAM $INPUT_IMAGE $@"
    $PROGRAM $INPUT_IMAGE "$@"
    echo "------------------------"
}


# Convolution examples
run_example convolution 16 16 sobel vertical
run_example convolution 16 16 sobel horizontal

# Flip examples
run_example flip 16 16 horizontal
run_example flip 16 16 vertical

# Grayscale example
run_example grayscale 16 16

# Blur example
run_example blur 16 16 5

