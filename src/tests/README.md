# rdv unit tests

Run everything:

```
pytest
```

(from the repo root - `pythonpath`/`testpaths` are already set in `pyproject.toml`, so no
install step or `PYTHONPATH` juggling is needed.)

Run only the tests that don't need a real GPU (fast - class/generics/metaclass logic only):

```
pytest -m "not gpu"
```

Run only the GPU-dependent ones (dispatches a shader or allocates a device buffer):

```
pytest -m gpu
```

## Layout

- `test_generics.py` / `test_aliases.py` / `test_dependency_order.py` / `test_cast.py::test_cast_*_generic*`
  - Pure class-creation/metaclass logic. No `@pytest.mark.gpu`: safe to run with no Vulkan
    device available at all.
- `test_cast.py` (the rest) / `test_tensor_reinterpret.py`
  - Marked `@pytest.mark.gpu`: construct real Node instances (allocates a small buffer) or
    dispatch a compute shader. Still fast (single dispatch, small data) - this is not where
    the sandbox/ benchmark scripts' 1000-iteration loops belong.

Sandbox scripts (`src/sandbox/*.py`) are examples/benchmarks meant to be run directly with
`python`, not test cases - they print/plot rather than assert, and some intentionally loop
thousands of times to measure throughput.
