# Contributing to Polygraphy

## Table of Contents

- [Design Principles](#design-principles)
- [Submitting Merge Requests](#submitting-merge-requests)


## Design Principles

### Amazing Error Messages

Error messages should ideally tell the user how to fix the error, or, failing that,
should try to make the cause of the error as obvious as possible. An overly verbose error
is better than a cryptic one.

### Simple But Flexible

The API should be as simple as possible, but also provide ways to completely customize
everything.
Loader composition is an example of this - advanced users can write their own loaders,
while still being able to use loaders provided by Polygraphy.

### None Means Default

Universally using `None` to indicate default value has some advantages:
- Makes it easier to write wrappers - instead of trying to match the default
    values of the function being wrapped, users can just use `None` .

- Can help prevent surprises caused by default value behavior in Python, as explained in
    the [comment for default_value()](./polygraphy/util/misc.py)

### Loader Naming Conventions

- Loaders that convert from a source format to some target format should
follow the naming convention: `<Target>From<Source>`, e.g. `OnnxFromTfGraph`, `NetworkFromOnnxBytes`

- Loaders that do not affect the format of their source should follow the naming convention:
`<Verb><Source>`, e.g. `ModifyOnnx`, `SaveEngine`

- For all other loaders, make sure the name is concise, but descriptive, e.g. `LoadPlugins`,
`CreateConfig`


## Submitting Merge Requests

1. Create a separate branch for your feature or bug fix
2. Make your changes
3. Run Tests:
    - Install prerequisite packages with:
        - `python3 -m pip install -r tests/requirements.txt`
        - `python3 -m pip install -r docs/requirements.txt`
    - Run tests with: `make test`
4. Commit, push, and submit a merge request to the main branch
