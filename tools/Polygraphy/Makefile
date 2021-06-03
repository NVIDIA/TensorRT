.PHONY: test test_noparallel clean build install_venv install docs

# Tests also check that docs can build
test: docs
	python3 -m pytest tests/ -v -x -nauto --dist=loadscope

test_noparallel: docs
	python3 -m pytest tests/ -v -x

clean:
	rm -rf dist/ build/ polygraphy.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_common: build
	-python3 -m pip install colored

install_venv: install_common
	python3 -m pip install $(CURDIR)/dist/*.whl
	python3 -m pip install --no-deps --force-reinstall $(CURDIR)/dist/*.whl

install: install_common
	python3 -m pip install $(CURDIR)/dist/*.whl --user
	python3 -m pip install --no-deps --force-reinstall $(CURDIR)/dist/*.whl --user

docs:
	mkdir -p build/docs
	sphinx-build docs build/docs/ -j auto -W
