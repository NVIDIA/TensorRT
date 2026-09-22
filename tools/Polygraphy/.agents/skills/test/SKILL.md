---
name: test
description: "Run Polygraphy tests with the correct environment setup. Activates when asked to run tests, run the test suite, or run a specific test file. Also use this proactively after making code changes — e.g. after a refactor, bug fix, or any edit that could affect runtime behaviour — to verify nothing is broken before committing."
---

# Run Polygraphy Tests

## Required environment variables

Set these before running any pytest command (from the Makefile `test` target):

- `PYTHONPATH` must include the repo root
- `POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1`
- `CUDA_MODULE_LOADING=LAZY`

## Running a specific test file or subset

```bash
PYTHONPATH="$(pwd):${PYTHONPATH}" \
POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1 \
CUDA_MODULE_LOADING=LAZY \
python3 -m pytest <path/to/test_file.py> -v --durations=15 --failed-first --new-first --script-launch-mode=subprocess
```

## Running the full test suite

Serial tests must run before parallel ones:

```bash
PYTHONPATH="$(pwd):${PYTHONPATH}" \
POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1 \
CUDA_MODULE_LOADING=LAZY \
python3 -m pytest tests/ -m "serial and not slow" -v --durations=15 --failed-first --new-first --script-launch-mode=subprocess && \
python3 -m pytest tests/ -n 8 --dist=loadscope -m "not serial and not slow" -v --durations=15 --failed-first --new-first --script-launch-mode=subprocess
```

## Notes

- Always pass `--script-launch-mode=subprocess` to avoid in-process runner issues with stdout/stderr capture.
- Skip `-n 8 --dist=loadscope` when running a small subset.
- `slow` tests are excluded by default; drop `and not slow` to include them.
- **Use this skill proactively** after any code change (refactor, bug fix, new feature) that could affect runtime behaviour — run the relevant test file before committing, without waiting for the user to ask.
