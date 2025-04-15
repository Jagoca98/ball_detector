#!/bin/bash

# This script is used to run the ukf_cpp process in a docker container

# Get the current directory
set -e
CURRENT_DIR=$(pwd)

#  Load the environment variables from the config.env file
set -o allexport
source .env
set -o allexport

# Ensure environment variables are set
: "${USER_ID:?Need to set USER_ID}"
: "${USER_NAME:?Need to set USER_NAME}"
: "${GROUP_ID:?Need to set GROUP_ID}"
: "${GROUP_NAME:?Need to set GROUP_NAME}"
: "${WORKSPACE:?Need to set WORKSPACE}"
: "${DOCKER_IMAGE_NAME:?Need to set DOCKER_IMAGE_NAME}"

        # -v ./data:/data/:ro \
        # -v ./data/output:/data/output/:rw \
# Generating the data inside a docker container
echo "Generating the data..."
if docker run \
        --name $DOCKER_IMAGE_NAME \
        -v ./$WORKSPACE:/$WORKSPACE/ \
        -u $USER_ID:$GROUP_ID \
        --device=/dev/dri:/dev/dri \
        --device=/dev/video0:/dev/video0 \
        --group-add video \
        --net=host \
        -e DISPLAY=$DISPLAY \
        -e TERM=xterm-256color \
        --rm \
        -it \
        $DOCKER_IMAGE_NAME \
        bash; then
    echo "Data generation successful."
else
    echo "Error: Data generation failed."
    exit 1
fi