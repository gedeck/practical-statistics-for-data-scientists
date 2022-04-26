PSFDS_IMAGE=psfds
JUPYTER_IMAGE=psfds_jupyter

jupyter:
	docker run --rm -v $(PWD):/src -p 8893:8893 $(JUPYTER_IMAGE) jupyter notebook --allow-root --port=8893 --ip 0.0.0.0 --no-browser

bash:
	docker run -it --rm -v $(PWD):/src $(PSFDS_IMAGE) bash

bash-jupyter:
	docker run -it --rm -v $(PWD):/src $(JUPYTER_IMAGE) bash

build: 
	docker build -t $(PSFDS_IMAGE) -f docker/Dockerfile.psfds .
	docker build -t $(JUPYTER_IMAGE) -f docker/Dockerfile.jupyter .

