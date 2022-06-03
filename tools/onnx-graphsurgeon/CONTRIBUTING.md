# Contributing To ONNX-GraphSurgeon

## Contributing

1. Create a separate branch for your feature or bug fix.
    You may want to create the branch on your own fork of the TensorRT OSS repository.

2. Make your changes and add corresponding tests.

    The structure of the `tests` directory closely mirrors that of the main source directory (`onnx_graphsurgeon`),
    so in general, for every source file you change, you'll need to modify the corresponding test file.

    When addiing new examples, be sure to add an entry in `test_examples.py`. The test will parse the README
    to execute any commands specified. If your example creates any additional files, specify each of them in
    the test case as an `Artifact`.

    If your changes are user-visible, make sure to update [CHANGELOG.md](CHANGELOG.md).

3. Run Tests:

    - Install prerequisite packages with:
        - `python3 -m pip install -r tests/requirements.txt`
        - `python3 -m pip install -r docs/requirements.txt`

    - Run tests with: `make test`

4. Commit, push, and submit a merge request.
