## docker.mk - Docker build targets
# This file is included by the main Makefile

# Declare phony targets (they don't produce files)
.PHONY: docker-build docker-run docker-clean

# Docker-specific variables
DOCKER_FOLDER := docker
DOCKER_IMAGE_NAME ?= $(shell basename $(CURDIR))

##@ Docker
docker-build: ## build Docker image
	@if [ ! -f "${DOCKER_FOLDER}/Dockerfile" ]; then \
	  printf "${YELLOW}[WARN] No ${DOCKER_FOLDER}/Dockerfile found, skipping build${RESET}\n"; \
	else \
	  printf "${BLUE}[INFO] Building Docker image with Python ${PYTHON_VERSION}${RESET}\n"; \
	  docker buildx build \
	    --file ${DOCKER_FOLDER}/Dockerfile \
	    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
	    --tag ${DOCKER_IMAGE_NAME}:latest \
	    --load \
	    .; \
	fi

docker-run: docker-build ## run the Docker container
	@printf "${BLUE}[INFO] Running Docker container ${DOCKER_IMAGE_NAME}${RESET}\n"
	@docker run --rm -it ${DOCKER_IMAGE_NAME}:latest

docker-clean: ## remove Docker image
	@printf "${BLUE}[INFO] Removing Docker image ${DOCKER_IMAGE_NAME}${RESET}\n"
	@docker rmi ${DOCKER_IMAGE_NAME}:latest 2>/dev/null || true
