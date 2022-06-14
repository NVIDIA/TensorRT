.PHONY: test leak_check clean build install docs

NPROC ?= 8

# Tests also check that docs can build
test: docs
    # Some tests need to be run serially - we annotate those with a `serial` marker.
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH} && \
		export PATH=$(CURDIR)/bin:$${PATH} && \
		export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1 && \
		python3 -m pytest tests/ -v -x --durations=5 -m "serial" && \
		\
		export PYTHONPATH=$(CURDIR):$${PYTHONPATH} && \
	    export PATH=$(CURDIR)/bin:$${PATH} && \
	    export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1 && \
	    python3 -m pytest tests/ -v -x -n $(NPROC) --dist=loadscope --durations=5 -m "not serial"

leak_check:
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH} && \
	    export PATH=$(CURDIR)/bin:$${PATH} && \
	    export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1 && \
		valgrind --leak-check=full python3 -m pytest tests/ -v --durations=5 2>&1 | tee leak-check.log

clean:
	rm -rf dist/ build/ polygraphy.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_deps: build
	-python3 -m pip install colored wheel

install: install_deps
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

docs: clean
	mkdir -p build/docs
	python3 `which sphinx-build` docs build/docs/ -j $(NPROC) -W
