# NVIDIA QAT toolkit for Tensorflow change log

Dates are in YYYY-MM-DD format.

## v0.2.0 (2022-09-09)
- Fixed bug in `infer_engine.py` (low accuracy due to indentation error).
- Added Inception-v3 support: code and results.
- Fixed User Guide links (they were the internal links, not public ones).
- Fixed bug in `examples/mobilenet/test_qdq_node_placement.py` (we implemented a more general `get_tfkeras_model` for Keras models QAT workflows, but missed that change in the MobileNet test script).
- Added Pillow requirement in examples.
- Added note for Conv2DTranspose support in the main README file.
- Improved generalization of residual branch detection for QDQ node placement.

## v0.1.0 (2022-06-17)

- Initial release of the toolkit.
