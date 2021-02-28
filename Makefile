SRC=src

# Django server
IMAGE=psfds
RUN=docker run -it --rm -v $(PWD):/code
RUN_IMAGE=$(RUN) $(IMAGE)

jupyter: docker/image.$(IMAGE)
	@ $(RUN) -p 8893:8893 $(IMAGE) jupyter notebook --allow-root --port=8893

bash:
	@ $(RUN_IMAGE) bash


# Docker container
build: touch-docker docker/image.$(IMAGE)

touch-docker:
	touch docker/Dockerfile.$(IMAGE)

docker/image.$(IMAGE): docker/Dockerfile.$(IMAGE)
	docker build -t $(IMAGE) -f docker/Dockerfile.$(IMAGE) .
	@ touch $@

