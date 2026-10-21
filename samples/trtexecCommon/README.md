# tools/trtexecCommon

Shared utility library (`trtexec_common`) backing `trtexec` and `trtexec_safe`.

`trtexec_safe` uses only `safeCommon.h` and `safeErrorRecorder.h`; everything else here is
trtexec's engine, option model and reporting.

The TensorRT samples live in the OSS repo and carry their own copy of the utilities they
need under `samples/common`.
