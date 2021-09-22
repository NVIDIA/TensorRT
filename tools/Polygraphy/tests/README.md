# Tests

The tests directory closely mirrors the structure of the main `polygraphy` directory.

## Adding Tests

For a given submodule, add tests into the corresponding directory under `tests/`.

## Parallel Test Execution

By default, the Polygraphy build system runs tests in parallel. However, some tests
may not be good candidates for parallel exection - for example, performance tests.
You can selectively disable parallel execution for these tests using the `pytest.mark.serial`
marker. The build system will exclude these tests from parallel execution and run them serially instead.

For example:
```python
@pytest.mark.serial
def my_not_parallel_test():
    ...
```
