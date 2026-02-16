SHELL := /bin/bash

.PHONY: ci test

ci:
	./scripts/ci.sh

test:
	./scripts/test.sh
