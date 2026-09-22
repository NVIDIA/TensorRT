# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance analyzer used by the plugin autotuner.
"""

import ctypes
import time
from typing import Dict, List, Any, Optional, Tuple

from polygraphy import mod
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
trt = lazy_import_trt()
cuda_runtime = mod.lazy_import(
    "cuda.bindings.runtime", pkg_name="cuda-python", log=False
)


class PerformanceMetrics:

    def __init__(self):
        self.latency_ms: float = 0.0
        self.throughput_fps: float = 0.0
        self.memory_usage_mb: float = 0.0
        self.accuracy: Optional[float] = None
        self.timestamp: float = 0.0

    def __str__(self):
        return (
            f"Latency: {self.latency_ms:.2f}ms, "
            f"Throughput: {self.throughput_fps:.2f}fps, "
            f"Memory: {self.memory_usage_mb:.2f}MB"
        )


class PerformanceAnalyzer:

    def __init__(
        self,
        builder: Optional[Any] = None,
        config: Optional[Any] = None,
        workspace_size: int = 1 << 30,
        trt_log_level: Optional[int] = None,
    ):
        # Lazily create builder/config if not provided
        if builder is None or config is None:
            from polygraphy.backend.trt import get_trt_logger

            if builder is None:
                builder = trt.Builder(get_trt_logger())
            if config is None:
                config = builder.create_builder_config()

        self.builder = builder
        self.config = config
        self.workspace_size = workspace_size
        self.trt_log_level = (
            trt_log_level if trt_log_level is not None else trt.Logger.INFO
        )
        self.num_iterations = 10
        self.warmup_iterations = 2
        self.total_compile_time = 0.0
        self.num_engine_builds = 0

    def _get_cuda_runtime(self):
        """Get CUDA runtime, loading it if necessary."""
        return cuda_runtime

    def _cuda_call(self, call):
        """Wrapper for CUDA runtime calls with error checking."""
        runtime = self._get_cuda_runtime()
        err, result = call[0], call[1:]
        if isinstance(err, runtime.cudaError_t):
            if err != runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA Runtime Error: {err}")
        else:
            raise RuntimeError(f"Unknown error type: {err}")
        if len(result) == 1:
            result = result[0]
        return result

    def analyze_network(
        self,
        network: "trt.INetworkDefinition",
        input_data: Optional[Dict[str, Any]] = None,
        num_iterations: Optional[int] = None,
        warmup_iterations: Optional[int] = None,
    ) -> PerformanceMetrics:
        if num_iterations is None:
            num_iterations = self.num_iterations
        if warmup_iterations is None:
            warmup_iterations = self.warmup_iterations
        return self._analyze_network_real_gpu(
            network, input_data, num_iterations, warmup_iterations
        )

    def _analyze_network_real_gpu(
        self,
        network: "trt.INetworkDefinition",
        input_data: Optional[Dict[str, Any]] = None,
        num_iterations: int = 10,
        warmup_iterations: int = 2,
    ) -> PerformanceMetrics:
        engine = self._build_engine(network)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        context = engine.create_execution_context()
        self.builder.logger.min_severity = self.trt_log_level

        if input_data is None:
            input_data = self._generate_dummy_inputs(engine)

        inputs, outputs, bindings = self._allocate_memory(engine, context, input_data)

        # Warmup
        for _ in range(warmup_iterations):
            try:
                self._run_inference(context, bindings)
            except Exception as e:
                G_LOGGER.warning(f"Warmup iteration failed: {e}")
                continue

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start_time = time.time()
            self._run_inference(context, bindings)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)

        metrics = PerformanceMetrics()
        if latencies:
            metrics.latency_ms = float(np.mean(latencies))
            metrics.throughput_fps = (
                1000.0 / metrics.latency_ms if metrics.latency_ms > 0 else 0.0
            )
            metrics.memory_usage_mb = self._estimate_memory_usage(engine)
        else:
            raise RuntimeError("No successful benchmark iterations completed")

        self._cleanup(engine, context, inputs, outputs, bindings)
        return metrics

    def _build_engine(
        self, network: "trt.INetworkDefinition"
    ) -> Optional["trt.ICudaEngine"]:
        compile_start = time.time()
        try:
            self.config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.workspace_size
            )
            self.builder.logger.min_severity = self.trt_log_level
            serialized = self.builder.build_serialized_network(network, self.config)
            if serialized is None:
                return None
            runtime = trt.Runtime(self.builder.logger)
            engine = runtime.deserialize_cuda_engine(serialized)
            self.total_compile_time += time.time() - compile_start
            self.num_engine_builds += 1
            return engine
        except Exception:
            self.total_compile_time += time.time() - compile_start
            self.num_engine_builds += 1
            return None

    def get_compile_time_stats(self) -> Tuple[float, int]:
        return self.total_compile_time, self.num_engine_builds

    def _generate_dummy_inputs(self, engine) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = tuple(engine.get_tensor_shape(name))
                dtype = self._trt_dtype_to_numpy(engine.get_tensor_dtype(name))
                inputs[name] = np.random.random(shape).astype(dtype)
        return inputs

    def _trt_dtype_to_numpy(self, trt_dtype) -> Any:
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.INT32: np.int32,
            trt.DataType.BOOL: np.bool_,
        }
        return mapping.get(trt_dtype, np.float32)

    def _allocate_memory(self, engine, context, input_data):
        inputs = []
        outputs = []
        bindings: List[int] = []
        runtime = self._get_cuda_runtime()

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = self._trt_dtype_to_numpy(engine.get_tensor_dtype(name))
            mode = engine.get_tensor_mode(name)

            size = int(np.prod(shape))
            num_bytes = size * dtype().itemsize
            host_mem = self._cuda_call(runtime.cudaMallocHost(num_bytes))
            pointer_type = ctypes.POINTER(ctypes.c_byte)
            host_ptr = ctypes.cast(host_mem, pointer_type)
            host_array = (
                np.ctypeslib.as_array(host_ptr, (size,)).reshape(shape).astype(dtype)
            )

            device_mem = self._cuda_call(runtime.cudaMalloc(num_bytes))
            bindings.append(int(device_mem))

            if mode == trt.TensorIOMode.INPUT:
                # Store original host_mem pointer for proper cleanup
                inputs.append(
                    {"host": host_array, "device": device_mem, "host_ptr": host_mem}
                )
                if name in input_data:
                    np.copyto(host_array, input_data[name])
            else:
                outputs.append(
                    {"host": host_array, "device": device_mem, "host_ptr": host_mem}
                )

        return inputs, outputs, bindings

    def _run_inference(self, context: "trt.IExecutionContext", bindings: List[int]):
        runtime = self._get_cuda_runtime()
        self.builder.logger.min_severity = self.trt_log_level
        stream = self._cuda_call(runtime.cudaStreamCreate())
        engine = context.engine
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            context.set_tensor_address(name, bindings[i])
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        self._cuda_call(runtime.cudaStreamSynchronize(stream))
        self._cuda_call(runtime.cudaStreamDestroy(stream))

    def _estimate_memory_usage(self, engine: "trt.ICudaEngine") -> float:
        total = 0
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = self._trt_dtype_to_numpy(engine.get_tensor_dtype(name))
            total += int(np.prod(shape)) * dtype().itemsize
        return total / (1024 * 1024)

    def _cleanup(self, engine, context, inputs, outputs, bindings):
        try:
            runtime = self._get_cuda_runtime()
        except RuntimeError:
            # If CUDA runtime is not available during cleanup, just skip GPU cleanup
            G_LOGGER.warning("CUDA runtime not available for cleanup")
            return

        for binding in bindings:
            if binding:
                try:
                    self._cuda_call(runtime.cudaFree(binding))
                except Exception as e:
                    G_LOGGER.warning(f"Failed to free device memory: {e}")

        for memory_list in [inputs, outputs]:
            for mem in memory_list:
                # Use the original host_ptr for proper cleanup
                if "host_ptr" in mem:
                    try:
                        self._cuda_call(runtime.cudaFreeHost(mem["host_ptr"]))
                    except Exception as e:
                        G_LOGGER.warning(f"Failed to free host memory: {e}")

        # Note: TensorRT engine and context are managed by Python's garbage collector.
        # They will be automatically cleaned up when references are released.
        # Explicitly calling __del__() is not recommended and can cause double-free issues.
