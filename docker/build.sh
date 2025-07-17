#!/bin/bash

IMAGE_NAME="fish-tracker"

CONTAINER_NAME="fish-tracker-container"

# DOCKER_RUN_OPTS=(
#   --rm
#   -v "$(pwd)":/app
#   -w /app
# )

echo "Build Docker image : $IMAGE_NAME"
docker build -f docker/Dockerfile -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
  echo "Failed"
  exit 1
fi


# docker run "${DOCKER_RUN_OPTS[@]}" "$IMAGE_NAME"
