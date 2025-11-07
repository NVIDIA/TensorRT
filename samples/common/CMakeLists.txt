# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_library(trt_samples_common STATIC)

target_sources(trt_samples_common PRIVATE
    argsParser.h
    BatchStream.h
    bfloat16.cpp
    bfloat16.h
    buffers.h
    common.h
    debugTensorWriter.cpp
    debugTensorWriter.h
    EntropyCalibrator.h
    ErrorRecorder.h
    getOptions.cpp
    getOptions.h
    getoptWin.h
    half.h
    logger.cpp
    logger.h
    logging.h
    parserOnnxConfig.h
    sampleConfig.h
    sampleDevice.cpp
    sampleDevice.h
    sampleEngines.cpp
    sampleEngines.h
    sampleEntrypoints.h
    sampleInference.cpp
    sampleInference.h
    sampleOptions.cpp
    sampleOptions.h
    sampleReporting.cpp
    sampleReporting.h
    sampleUtils.cpp
    sampleUtils.h
    streamReader.h
)

if (MSVC)
    enable_language(C)
    target_sources(trt_samples_common PRIVATE
        getOpt.c
    )
endif()

target_include_directories(trt_samples_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)


target_link_libraries(trt_samples_common PUBLIC
    trt_global_definitions
    trt_shared
    Threads::Threads
    dl
    TRT::cudart
    $<COMPILE_ONLY:TRT_SAMPLES::tensorrt> # Each sample individually must determine its linkage to TRT.
    TRT_SAMPLES::onnxparser
)

# For statically-linked samples, we need to upgrade the link to always link TRT rather than letting the samples decide.
if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    target_link_libraries(trt_samples_common PUBLIC
        $<LINK_LIBRARY:WHOLE_ARCHIVE,TRT_SAMPLES::tensorrt> # Has to be whole archive so we keep the builder resources correctly.
    )
endif()


if(${TRT_BUILD_ENABLE_DLA})
    target_link_libraries(trt_samples_common PUBLIC NVDLA::compiler)
endif()
