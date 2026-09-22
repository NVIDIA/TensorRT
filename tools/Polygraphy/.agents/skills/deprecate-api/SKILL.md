---
name: deprecate-api
description: "Mark a Polygraphy function, class, module, or alias as deprecated so it warns at runtime and is scheduled for removal. Use when asked to deprecate an API, replace one API with another while keeping backwards compatibility, or add a deprecation warning."
---

# Deprecating Polygraphy APIs

Polygraphy provides deprecation helpers in `polygraphy/mod/exporter.py`, exposed via the `mod` module (`from polygraphy import mod`).

## Decorator: `@mod.deprecate` (preferred for whole functions/classes/modules)

```python
from polygraphy import mod

class CompareFunc:
    @staticmethod
    @mod.deprecate(
        remove_in="0.60.0",
        use_instead="SimpleCompareFunc",
        name="CompareFunc.simple",   # optional; defaults to obj.__name__
    )
    def simple(...):
        """<full docstring is preserved>"""
        return SimpleCompareFunc(...)
```

- `remove_in` (required): the version the symbol will be removed in. **Must be greater than the current `polygraphy.__version__`** (see `polygraphy/__init__.py`) — otherwise `INTERNAL_CORRECTNESS_CHECKS` raises an internal error at import. Recent deprecations target `"0.60.0"`.
- `use_instead`: the replacement to recommend (string). Pass `None` if there is no replacement.
- `name` / `module_name`: control the displayed name. For a `@staticmethod`, pass `name="ClassName.method"` since `obj.__name__` would otherwise just be the bare method name.
- **Decorator order matters**: put `@staticmethod` (or `@classmethod`) *above* `@mod.deprecate` so the static/class wrapper stays outermost and `mod.deprecate` receives the raw function.
- The decorator works on functions, classes (returns a subclass that warns on `__init__`), and modules.
- The decorator **preserves the original docstring** and prepends a `Deprecated: Use <X> instead.` note (see `deprecate` in `exporter.py`). This keeps API reference docs intact.

## Inline call: `mod.warn_deprecated` (for partial / conditional deprecations)

When only a specific argument value or code path is deprecated (not the whole function), call it directly:

```python
mod.warn_deprecated(
    "Returning NumPy dtypes from get_input_metadata()",
    use_instead=None,
    remove_in="0.60.0",
)
```

Examples in-tree: `polygraphy/backend/base/runner.py`, `polygraphy/backend/trt/loader.py`, `polygraphy/tools/args/util/util.py`.

## Deprecated aliases

To expose an old name that forwards to a new one, use `mod.export_deprecated_alias("OldName", remove_in="0.60.0", use_instead=None)` (see `export_deprecated_alias` in `exporter.py`). For an entire module: `mod.export_deprecated_alias("old_mod", remove_in="0.60.0")(sys.modules[__name__])`.

## Behavior & gotchas

- `warn_deprecated` emits a Python `DeprecationWarning` (hidden by default; run Python with `-W always::DeprecationWarning` to see it). Pass `always_show_warning=True` to also log it via `G_LOGGER.warning`.
- **Avoid making the library warn at itself**: update internal callers (and CLI script generation) to use the *new* API so normal usage doesn't trigger the deprecation. E.g. `Comparator.compare_accuracy` defaults to `SimpleCompareFunc()` rather than the deprecated `CompareFunc.simple()`.
- Tests that still call the deprecated API will pass (there is no `filterwarnings = error` config) but will emit warnings.

## After deprecating — verify

```bash
PYTHONPATH=<repo> python3 -W always::DeprecationWarning -c "
import warnings
from polygraphy.comparator import CompareFunc
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    CompareFunc.simple()
assert any(issubclass(x.category, DeprecationWarning) for x in w)
print('docstring preserved:', 'Args:' in (CompareFunc.simple.__doc__ or ''))
"
```

If you added a *new* lazy dependency at module scope while doing this, also update the expected set in `tests/mod/test_dependencies.py::TestDependencies::test_all_lazy_imports`.
