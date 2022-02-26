# Contributing to Polygraphy

## Table of Contents

- [Design Principles](#design-principles)
- [Deprecation Scheme](#deprecation-scheme)
- [Submitting Merge Requests](#submitting-merge-requests)


## Design Principles

### Amazing Error Messages

Error messages should ideally tell the user how to fix the error, or, failing that,
should try to make the cause of the error as obvious as possible. An overly verbose error
is better than a cryptic one.

### Simple But Flexible

The API should be as simple as possible, with plug-and-play modular components.
Loader composition is an example of this - users can freely intermix Polygraphy's
loaders with backend APIs. See [example 03](examples/api/03_interoperating_with_tensorrt/).

### None Means Default

Universally using `None` to indicate default value has some advantages:
- Makes it easier to write wrappers - instead of trying to match the default
    values of the function being wrapped, users can just use `None` .

- Can help prevent surprises caused by default value behavior in Python, as explained in
    the [comment for default()](./polygraphy/util/util.py)

### Descriptive Loader Names

- Loaders that convert from a source format to some target format should
follow the naming convention: `<Target>From<Source>`, e.g. `OnnxFromTfGraph`, `NetworkFromOnnxBytes`

- Loaders that do not affect the format of their source should follow the naming convention:
`<Verb><Source>`, e.g. `ModifyOutputs`, `SaveEngine`

- For all other loaders, make sure the name is concise, but descriptive, e.g. `LoadPlugins`,
`CreateConfig`


## Deprecation Scheme

### Annotating Classes And Functions

To indicate that a class or function is deprecated, you can decorate it
with the `deprecate()` decorator defined in `exporter.py`. For example:

```python
@mod.deprecate(remove_in="0.25.0", use_instead="NewClass")
class OldClass(object):
    ...
```

When the decorated type is used, a `DeprecationWarning` will be issued.

### Exporting Aliases

In some cases, it may be necessary to rename a function, class, or module.
In those cases, we can export the old name as a deprecated alias to preserve backwards compatibility.

- For a class or function, annotate the replacement with the `export_deprecated_alias` decorator.
    For example:

    ```python
    @mod.export_deprecated_alias("Old", remove_in="0.25.0")
    class New(object):
        ...
    ```

- For modules, invoke the decorator manually within the module file.
    For example:

    ```python
    mod.export_deprecated_alias("old_mod_name", remove_in="0.25.0")(sys.modules[__name__])
    ```

### Adding Tests

When you deprecate an API, be sure to add a test into `tests/test_deprecated_aliases.py`
for the deprecated type.
The tests there will automatically fail if the deprecated type is not removed in the version
specified in `remove_in`.


## Submitting Merge Requests

1. Create a separate branch for your feature or bug fix
2. Make your changes
3. Run Tests:
    - Install prerequisite packages with:
        - `python3 -m pip install -r tests/requirements.txt --index-url https://pypi.ngc.nvidia.com`
        - `python3 -m pip install -r docs/requirements.txt --index-url https://pypi.ngc.nvidia.com`
    - Run tests with: `make test`
4. Commit, push, and submit a merge request to the main branch
