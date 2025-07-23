/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This contains the fundamental types, i.e. Dims, Weights, dtype
#include "ForwardDeclarations.h"
#include "utils.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "infer/pyFoundationalTypesDoc.h"
#include <cuda_runtime_api.h>

namespace tensorrt
{
using namespace nvinfer1;

namespace lambdas
{
// For Weights
static const auto weights_datatype_constructor = [](DataType const& type) { return new Weights{type, nullptr, 0}; };

static const auto weights_pointer_constructor = [](DataType const& type, size_t const ptr, int64_t count) {
    return new Weights{type, reinterpret_cast<void*>(ptr), count};
};

static const auto weights_numpy_constructor = [](py::array& arr) {
    arr = py::array::ensure(arr);
    // In order to construct a weights object, we must have a contiguous C-style array.
    PY_ASSERT_VALUE_ERROR(
        arr, "Could not convert NumPy array to Weights. Is it using a data type supported by TensorRT?");
    PY_ASSERT_VALUE_ERROR((arr.flags() & py::array::c_style),
        "Could not convert non-contiguous NumPy array to Weights. Please use numpy.ascontiguousarray() to fix this.");
    return new Weights{utils::type(arr.dtype()), arr.data(), arr.size()};
};

// Helper to compare dims with any kind of Python Iterable.
template <typename DimsType, typename PyIterable>
bool dimsEqual(DimsType const& self, PyIterable& other)
{
    if (static_cast<int64_t>(other.size()) != self.nbDims)
    {
        return false;
    }
    bool eq = true;
    std::vector<int64_t> o = other.template cast<std::vector<int64_t>>();
    for (int32_t i = 0; i < self.nbDims; ++i)
    {
        eq = eq && (self.d[i] == o[i]);
    }
    return eq;
}

// For base Dims class
static const auto dims_vector_constructor = [](std::vector<int64_t> const& in) {
    // This is required, because otherwise MAX_DIMS will not be resolved at compile time.
    int32_t const maxDims{static_cast<int32_t>(Dims::MAX_DIMS)};
    PY_ASSERT_VALUE_ERROR(in.size() <= maxDims,
        "Input length " + std::to_string(in.size()) + ". Max expected length is " + std::to_string(maxDims));

    // Create the Dims object.
    Dims* self = new Dims{};
    self->nbDims = in.size();
    for (int32_t i = 0; static_cast<size_t>(i) < in.size(); ++i)
        self->d[i] = in[i];
    return self;
};

static const auto dims_to_str = [](Dims const& self) {
    if (self.nbDims == 0)
        return std::string("()");
    // Length 1 should followed by trailing comma, for tuple-like behavior.
    if (self.nbDims == 1)
        return "(" + std::to_string(self.d[0]) + ",)";
    // Non-zero lengths
    std::string temp = "(";
    for (int32_t i = 0; i < self.nbDims - 1; ++i)
        temp += std::to_string(self.d[i]) + ", ";
    temp += std::to_string(self.d[self.nbDims - 1]) + ")";
    return temp;
};

static const auto dims_len = [](Dims const& self) { return self.nbDims; };

// TODO: Add slicing support?
static const auto dims_getter = [](Dims const& self, int32_t const pyIndex) -> int64_t const& {
    // Without these bounds checks, horrible infinite looping will occur.
    int32_t const index{(pyIndex < 0) ? static_cast<int32_t>(self.nbDims) + pyIndex : pyIndex};
    PY_ASSERT_INDEX_ERROR(index >= 0 && index < self.nbDims);
    return self.d[index];
};

static const auto dims_getter_slice = [](Dims const& self, py::slice slice) {
    int64_t start, stop, step, slicelength;
    PY_ASSERT_VALUE_ERROR(
        slice.compute(self.nbDims, &start, &stop, &step, &slicelength), "Incorrect getter slice dims");
    // Disallow out-of-bounds things.
    PY_ASSERT_INDEX_ERROR(stop <= self.nbDims);

    py::tuple ret{slicelength};
    for (int32_t i = start, index = 0; i < stop; i += step, ++index)
        ret[index] = self.d[i];
    return ret;
};

static const auto dims_setter = [](Dims& self, int32_t const pyIndex, int64_t const item) {
    int32_t const index{(pyIndex < 0) ? static_cast<int32_t>(self.nbDims) + pyIndex : pyIndex};
    PY_ASSERT_INDEX_ERROR(index >= 0 && index < self.nbDims);
    self.d[index] = item;
};

static const auto dims_setter_slice = [](Dims& self, py::slice slice, Dims const& other) {
    int64_t start, stop, step, slicelength;
    PY_ASSERT_VALUE_ERROR(
        slice.compute(self.nbDims, &start, &stop, &step, &slicelength), "Incorrect setter slice dims");
    // Disallow out-of-bounds things.
    PY_ASSERT_INDEX_ERROR(stop < self.nbDims);

    for (int32_t i = start, index = 0; i < stop; i += step, ++index)
        self.d[i] = other.d[index];
};

// For Dims2
static const auto dims2_vector_constructor = [](std::vector<int64_t> const& in) {
    PY_ASSERT_VALUE_ERROR(in.size() == 2,
        "Input length " + std::to_string(in.size()) + " not equal to expected Dims2 length, which is 2");
    return new Dims2{in[0], in[1]};
};

// For DimsHW
static const auto dimshw_vector_constructor = [](std::vector<int64_t> const& in) {
    PY_ASSERT_VALUE_ERROR(in.size() == 2,
        "Input length " + std::to_string(in.size()) + " not equal to expected DimsHW length, which is 2");
    return new DimsHW{in[0], in[1]};
};

// For Dims3
static const auto dims3_vector_constructor = [](std::vector<int64_t> const& in) {
    PY_ASSERT_VALUE_ERROR(in.size() == 3,
        "Input length " + std::to_string(in.size()) + " not equal to expected Dims3 length, which is 3");
    return new Dims3{in[0], in[1], in[2]};
};

// For Dims4
static const auto dims4_vector_constructor = [](std::vector<int64_t> const& in) {
    PY_ASSERT_VALUE_ERROR(in.size() == 4,
        "Input length " + std::to_string(in.size()) + " not equal to expected Dims4 length, which is 4");
    return new Dims4{in[0], in[1], in[2], in[3]};
};

// For IHostMemory
static const auto host_memory_buffer_interface = [](IHostMemory& self) -> py::buffer_info {
    return py::buffer_info(self.data(),         /* Pointer to buffer */
        utils::size(self.type()),               /* Size of one scalar */
        py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
        1,                                      /* Number of dimensions */
        {self.size()},                          /* Buffer dimensions */
        {utils::size(self.type())}              /* Strides (in bytes) for each index */
    );
};
} // namespace lambdas

void bindFoundationalTypes(py::module& m)
{
    // Bind the top level DataType enum.
    py::enum_<DataType>(m, "DataType", DataTypeDoc::descr, py::module_local())
        .value("FLOAT", DataType::kFLOAT, DataTypeDoc::float32)
        .value("HALF", DataType::kHALF, DataTypeDoc::float16)
        .value("BF16", DataType::kBF16, DataTypeDoc::bfloat16)
        .value("INT8", DataType::kINT8, DataTypeDoc::int8)
        .value("INT32", DataType::kINT32, DataTypeDoc::int32)
        .value("INT64", DataType::kINT64, DataTypeDoc::int64)
        .value("BOOL", DataType::kBOOL, DataTypeDoc::boolean)
        .value("UINT8", DataType::kUINT8, DataTypeDoc::uint8)
        .value("FP8", DataType::kFP8, DataTypeDoc::fp8)
        .value("INT4", DataType::kINT4, DataTypeDoc::int4)
        .value("FP4", DataType::kFP4, DataTypeDoc::fp4)
        .value("E8M0", DataType::kE8M0, DataTypeDoc::e8m0); // DataType

    // Also create direct mappings (so we can call trt.float32, for example).
    m.attr("float32") = DataType::kFLOAT;
    m.attr("float16") = DataType::kHALF;
    m.attr("bfloat16") = DataType::kBF16;
    m.attr("int8") = DataType::kINT8;
    m.attr("int32") = DataType::kINT32;
    m.attr("int64") = DataType::kINT64;
    m.attr("bool") = DataType::kBOOL;
    m.attr("uint8") = DataType::kUINT8;
    m.attr("fp8") = DataType::kFP8;
    m.attr("int4") = DataType::kINT4;
    m.attr("fp4") = DataType::kFP4;
    m.attr("e8m0") = DataType::kE8M0;

    py::enum_<WeightsRole>(m, "WeightsRole", WeightsRoleDoc::descr, py::module_local())
        .value("KERNEL", WeightsRole::kKERNEL, WeightsRoleDoc::KERNEL)
        .value("BIAS", WeightsRole::kBIAS, WeightsRoleDoc::BIAS)
        .value("SHIFT", WeightsRole::kSHIFT, WeightsRoleDoc::SHIFT)
        .value("SCALE", WeightsRole::kSCALE, WeightsRoleDoc::SCALE)
        .value("CONSTANT", WeightsRole::kCONSTANT, WeightsRoleDoc::CONSTANT)
        .value("ANY", WeightsRole::kANY, WeightsRoleDoc::ANY); // WeightsRole

    // Weights
    py::class_<Weights>(m, "Weights", WeightsDoc::descr, py::module_local())
        // Can construct an empty weights object with type. Defaults to float32.
        .def(py::init(lambdas::weights_datatype_constructor), "type"_a = DataType::kFLOAT, WeightsDoc::init_type)
        .def(py::init(lambdas::weights_pointer_constructor), "type"_a, "ptr"_a, "count"_a, WeightsDoc::init_ptr)
        // Allows for construction through any contiguous numpy array. It then keeps a pointer to that buffer
        // (zero-copy).
        .def(py::init(lambdas::weights_numpy_constructor), "a"_a, py::keep_alive<1, 2>(), WeightsDoc::init_numpy)
        // Expose numpy-like attributes.
        .def_property_readonly("dtype", [](Weights const& self) -> DataType { return self.type; })
        .def_property_readonly("size", [](Weights const& self) { return self.count; })
        .def_property_readonly("nbytes", [](Weights const& self) { return utils::size(self.type) * self.count; })
        .def("numpy", utils::weights_to_numpy, py::return_value_policy::reference_internal, WeightsDoc::numpy)
        .def("__len__", [](Weights const& self) { return static_cast<size_t>(self.count); }); // Weights

    // Also allow implicit construction, so we can pass in numpy arrays instead of Weights.
    py::implicitly_convertible<py::array, Weights>();

    // Dims
    py::class_<Dims>(m, "Dims", DimsDoc::descr, py::module_local())
        .def(py::init<>())
        // Allows for construction from python lists and tuples.
        .def(py::init(lambdas::dims_vector_constructor), "shape"_a)
        // static_cast is required here, or MAX_DIMS does not get pulled in until LOAD time.
        .def_property_readonly_static(
            "MAX_DIMS", [](py::object /*self*/) { return static_cast<int32_t>(Dims::MAX_DIMS); }, DimsDoc::MAX_DIMS)
        // Allow for string representations (displays like a python tuple).
        .def("__str__", lambdas::dims_to_str)
        .def("__repr__", lambdas::dims_to_str)
        // Allow direct comparisons with tuples and lists.
        .def("__eq__", lambdas::dimsEqual<Dims, py::list>)
        .def("__eq__", lambdas::dimsEqual<Dims, py::tuple>)
        // These functions allow us to use Dims like an iterable.
        .def("__len__", lambdas::dims_len)
        .def("__getitem__", lambdas::dims_getter)
        .def("__getitem__", lambdas::dims_getter_slice)
        .def("__setitem__", lambdas::dims_setter)
        .def("__setitem__", lambdas::dims_setter_slice); // Dims

    // Make it possible to use tuples/lists in Python in place of Dims.
    py::implicitly_convertible<std::vector<int64_t>, Dims>();

    // 2D
    py::class_<Dims2, Dims>(m, "Dims2", Dims2Doc::descr, py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t>(), "dim0"_a, "dim1"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dims2_vector_constructor), "shape"_a); // Dims2

    py::implicitly_convertible<std::vector<int64_t>, Dims2>();

    py::class_<DimsHW, Dims2>(m, "DimsHW", DimsHWDoc::descr, py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t>(), "h"_a, "w"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dimshw_vector_constructor), "shape"_a)
        // Expose these functions as attributes in Python.
        .def_property(
            "h", [](DimsHW const& dims) { return dims.h(); }, [](DimsHW& dims, int64_t i) { dims.h() = i; })
        .def_property(
            "w", [](DimsHW const& dims) { return dims.w(); }, [](DimsHW& dims, int64_t i) { dims.w() = i; }); // DimsHW

    py::implicitly_convertible<std::vector<int64_t>, DimsHW>();

    // 3D
    py::class_<Dims3, Dims>(m, "Dims3", Dims3Doc::descr, py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), "dim0"_a, "dim1"_a, "dim2"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dims3_vector_constructor), "shape"_a); // Dims3

    py::implicitly_convertible<std::vector<int64_t>, Dims3>();

    // 4D
    py::class_<Dims4, Dims>(m, "Dims4", Dims4Doc::descr, py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(), "dim0"_a, "dim1"_a, "dim2"_a, "dim3"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dims4_vector_constructor), "shape"_a); // Dims4

    py::implicitly_convertible<std::vector<int64_t>, Dims4>();

    py::class_<IHostMemory>(m, "IHostMemory", py::buffer_protocol(), IHostMemoryDoc::descr, py::module_local())
        .def_property_readonly("dtype", [](IHostMemory const& mem) { return mem.type(); })
        .def_property_readonly("nbytes", [](IHostMemory const& mem) { return mem.size(); })
        // Expose buffer interface.
        .def_buffer(lambdas::host_memory_buffer_interface)
        .def("__del__", &utils::doNothingDel<IHostMemory>); // IHostMemory

    py::class_<InterfaceInfo>(m, "InterfaceInfo", InterfaceInfoDoc::descr, py::module_local())
        .def_readwrite("kind", &InterfaceInfo::kind)
        .def_readwrite("major", &InterfaceInfo::major)
        .def_readwrite("minor", &InterfaceInfo::minor);

    py::enum_<APILanguage>(m, "APILanguage", APILanguageDoc::descr, py::module_local())
        .value("CPP", APILanguage::kCPP)
        .value("PYTHON", APILanguage::kPYTHON);

    py::class_<IVersionedInterface>(m, "IVersionedInterface", IVersionedInterfaceDoc::descr, py::module_local())
        .def_property_readonly("api_language", &IVersionedInterface::getAPILanguage)
        .def_property_readonly("interface_info", &IVersionedInterface::getInterfaceInfo);
}

} // namespace tensorrt
