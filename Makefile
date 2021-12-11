NAME?=deep-chimpact
COMMAND?=bash


.PHONY: build
build:
	docker build \
	-t $(NAME) -f Dockerfile .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: logs
logs:
	docker logs -f $(NAME)

.PHONY: exec
exec:
	docker exec -it $(NAME) bash

.PHONY: run-dev
run-dev:
	docker run -it -d --rm \
		--gpus=all \
		--mount type=bind,src="$(shell pwd)",dst=/workdir \
		--name=$(NAME) \
		$(NAME) \
		$(COMMAND)

.PHONY: lint
lint:
	docker run --rm \
		--mount type=bind,src=$(shell pwd),dst=/workdir \
		--name=$(NAME) \
		$(NAME) \
		mypy src; flake8 src; pylint src
