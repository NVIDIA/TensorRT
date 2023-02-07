## Scripts for TRT Deployment
For both baseline and QAT, change:
- `RESNET_DEPTH` for 50 or 101,
- `RESNET_VERSION` for v1 or v2,
- `BS` for which batch sizes you wish to evaluate the engine on.

#### Baseline
```
./scripts/deploy_engine_baseline.sh
```
> Change `ROOT_DIR` to where your ONNX file is.

#### QAT
```
./scripts/deploy_engine_qat.sh
```
> Change `QAT_SUBDIR` and `ROOT_DIR` to where your ONNX file is.

### Only accuracy
```
./scripts/infer_engine.sh
```
