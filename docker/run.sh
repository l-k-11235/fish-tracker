#!/bin/bash

IMAGE_NAME="fish-tracker"

CONTAINER_NAME="fish-tracker-container"

DOCKER_RUN_OPTS=(
  --rm

)

if [ $? -ne 0 ]; then
  echo "Failed"
  exit 1
fi
echo docker run "${DOCKER_RUN_OPTS[@]}" "$IMAGE_NAME"
docker run "${DOCKER_RUN_OPTS[@]}" "$IMAGE_NAME"
