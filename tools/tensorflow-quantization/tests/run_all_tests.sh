#!/bin/bash

# clean
rm -rf wrappers_test_saved_models
rm -rf quantize_model_test_saved_models
rm -rf utils_test_saved_models
rm -rf qdq_test_saved_models
rm -rf __pycache__
rm -rf custom_qdq_models

# Run quantize_config tests
python -m pytest quantize_config_test.py -rP

# Run QDQ insertion tests
python -m pytest quantize_qdq_insertion_test.py -rP

# Run wrappers tests
python -m pytest quantize_wrappers_test.py -rP

# Run wrappers base tests
python -m pytest quantize_wrapper_base_test.py -rP

# Run end to end training test
python -m pytest quantize_test.py -rP

# Run special qdq insertion tests
python -m pytest custom_qdq_cases_test.py -rP

# Run utils test
python -m pytest utils_test.py -rP