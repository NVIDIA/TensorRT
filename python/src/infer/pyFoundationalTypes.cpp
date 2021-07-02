/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
static const auto weights_datatype_constructor = [](const DataType& type) { return new Weights{type, nullptr, 0}; };

static const auto weights_numpy_constructor = [](py::array& arr) {
    // In order to construct a weights object, we must have a contiguous C-style array.
    arr = py::array::ensure(arr, py::array::c_style);
    if (!arr)
    {
        constexpr const char* err
            = "Cannot construct Weights object from non-contiguous array. Please use numpy.ascontiguousarray() "
              "to fix this.";
        std::cout << "[ERROR] " << err << std::endl;
        throw std::invalid_argument{err};
    }
    return new Weights{utils::type(arr.dtype()), arr.data(), arr.size()};
};

// Helper to compare dims with any kind of Python Iterable.
template <typename DimsType, typename PyIterable>
bool dimsEqual(const DimsType& self, PyIterable& other)
{
    if (other.size() != self.nbDims)
    {
        return false;
    }
    bool eq = true;
    std::vector<int> o = other.template cast<std::vector<int>>();
    for (int i = 0; i < self.nbDims; ++i)
    {
        eq = eq && (self.d[i] == o[i]);
    }
    return eq;
}

// For base Dims class
static const auto dims_vector_constructor = [](const std::vector<int>& in) {
    // This is required, because otherwise MAX_DIMS will not be resolved at compile time.
    const int maxDims = static_cast<const int>(Dims::MAX_DIMS);
    if (in.size() > maxDims || in.size() < 0)
        throw std::length_error(
            "Input length " + std::to_string(in.size()) + ". Max expected length is " + std::to_string(maxDims));

    // Create the Dims object.
    Dims* self = new Dims{};
    self->nbDims = in.size();
    for (int i = 0; i < in.size(); ++i)
        self->d[i] = in[i];
    return self;
};

static const auto dims_to_str = [](const Dims& self) {
    if (self.nbDims == 0)
        return std::string("()");
    // Length 1 should followed by trailing comma, for tuple-like behavior.
    if (self.nbDims == 1)
        return "(" + std::to_string(self.d[0]) + ",)";
    // Non-zero lengths
    std::string temp = "(";
    for (int i = 0; i < self.nbDims - 1; ++i)
        temp += std::to_string(self.d[i]) + ", ";
    temp += std::to_string(self.d[self.nbDims - 1]) + ")";
    return temp;
};

static const auto dims_len = [](const Dims& self) { return self.nbDims; };

// TODO: Add slicing support?
static const auto dims_getter = [](const Dims& self, int pyIndex) -> const int& {
    // Without these bounds checks, horrible infinite looping will occur.
    size_t index = (pyIndex < 0) ? static_cast<int>(self.nbDims) + pyIndex : pyIndex;
    if (index >= self.nbDims)
        throw py::index_error();
    return self.d[index];
};

static const auto dims_getter_slice = [](const Dims& self, py::slice slice) {
    size_t start, stop, step, slicelength;
    if (!slice.compute(self.nbDims, &start, &stop, &step, &slicelength))
        throw py::error_already_set();
    // Disallow out-of-bounds things.
    if (stop > self.nbDims)
        throw py::index_error();

    py::tuple ret{slicelength};
    for (int i = start, index = 0; i < stop; i += step, ++index)
        ret[index] = self.d[i];
    return ret;
};

static const auto dims_setter = [](Dims& self, int pyIndex, int item) {
    size_t index = (pyIndex < 0) ? static_cast<int>(self.nbDims) + pyIndex : pyIndex;
    if (index >= self.nbDims)
        throw py::index_error();
    self.d[index] = item;
};

static const auto dims_setter_slice = [](Dims& self, py::slice slice, const Dims& other) {
    size_t start, stop, step, slicelength;
    if (!slice.compute(self.nbDims, &start, &stop, &step, &slicelength))
        throw py::error_already_set();
    // Disallow out-of-bounds things.
    if (stop >= self.nbDims)
        throw py::index_error();

    for (int i = start, index = 0; i < stop; i += step, ++index)
        self.d[i] = other.d[index];
};

// For Dims2
static const auto dims2_vector_constructor = [](const std::vector<int>& in) {
    if (in.size() != 2)
        throw std::length_error(
            "Input length " + std::to_string(in.size()) + " not equal to expected Dims2 length, which is 2");
    return new Dims2{in[0], in[1]};
};

// For DimsHW
static const auto dimshw_vector_constructor = [](const std::vector<int>& in) {
    if (in.size() != 2)
        throw std::length_error(
            "Input length " + std::to_string(in.size()) + " not equal to expected DimsHW length, which is 2");
    return new DimsHW{in[0], in[1]};
};

// For Dims3
static const auto dims3_vector_constructor = [](const std::vector<int>& in) {
    if (in.size() != 3)
        throw std::length_error(
            "Input length " + std::to_string(in.size()) + " not equal to expected Dims3 length, which is 3");
    return new Dims3{in[0], in[1], in[2]};
};

// For Dims4
static const auto dims4_vector_constructor = [](const std::vector<int>& in) {
    if (in.size() != 4)
        throw std::length_error(
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
    py::enum_<DataType>(m, "DataType", DataTypeDoc::descr)
        .value("FLOAT", DataType::kFLOAT, DataTypeDoc::float32)
        .value("HALF", DataType::kHALF, DataTypeDoc::float16)
        .value("INT8", DataType::kINT8, DataTypeDoc::int8)
        .value("INT32", DataType::kINT32, DataTypeDoc::int32)
        .value("BOOL", DataType::kBOOL, DataTypeDoc::boolean); // DataType

    // Also create direct mappings (so we can call trt.float32, for example).
    m.attr("float32") = DataType::kFLOAT;
    m.attr("float16") = DataType::kHALF;
    m.attr("int8") = DataType::kINT8;
    m.attr("int32") = DataType::kINT32;
    m.attr("bool") = DataType::kBOOL;

    py::enum_<WeightsRole>(m, "WeightsRole", WeightsRoleDoc::descr)
        .value("KERNEL", WeightsRole::kKERNEL, WeightsRoleDoc::KERNEL)
        .value("BIAS", WeightsRole::kBIAS, WeightsRoleDoc::BIAS)
        .value("SHIFT", WeightsRole::kSHIFT, WeightsRoleDoc::SHIFT)
        .value("SCALE", WeightsRole::kSCALE, WeightsRoleDoc::SCALE)
        .value("CONSTANT", WeightsRole::kCONSTANT, WeightsRoleDoc::CONSTANT)
        .value("ANY", WeightsRole::kANY, WeightsRoleDoc::ANY); // WeightsRole

    // Weights
    py::class_<Weights>(m, "Weights", WeightsDoc::descr)
        // Can construct an empty weights object with type. Defaults to float32.
        .def(py::init(lambdas::weights_datatype_constructor), "type"_a = DataType::kFLOAT, WeightsDoc::init_type)
        // Allows for construction through any contiguous numpy array. It then keeps a pointer to that buffer
        // (zero-copy).
        .def(py::init(lambdas::weights_numpy_constructor), "a"_a, py::keep_alive<1, 2>(), WeightsDoc::init_numpy)
        // Expose numpy-like attributes.
        .def_property_readonly("dtype", [](const Weights& self) -> DataType { return self.type; })
        .def_property_readonly("size", [](const Weights& self) { return self.count; })
        .def_property_readonly("nbytes", [](const Weights& self) { return utils::size(self.type) * self.count; })
        .def("numpy", utils::weights_to_numpy, py::return_value_policy::reference_internal, WeightsDoc::numpy)
        .def("__len__", [](const Weights& self) { return static_cast<size_t>(self.count); }); // Weights

    // Also allow implicit construction, so we can pass in numpy arrays instead of Weights.
    py::implicitly_convertible<py::array, Weights>();

    // Dims
    py::class_<Dims>(m, "Dims", DimsDoc::descr)
        .def(py::init<>())
        // Allows for construction from python lists and tuples.
        .def(py::init(lambdas::dims_vector_constructor), "shape"_a)
        // static_cast is required here, or MAX_DIMS does not get pulled in until LOAD time.
        .def_property_readonly(
            "MAX_DIMS", [](const Dims& self) { return static_cast<const int>(self.MAX_DIMS); }, DimsDoc::MAX_DIMS)
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
    py::implicitly_convertible<std::vector<int>, Dims>();

    // 2D
    py::class_<Dims2, Dims>(m, "Dims2", Dims2Doc::descr)
        .def(py::init<>())
        .def(py::init<int, int>(), "dim0"_a, "dim1"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dims2_vector_constructor), "shape"_a); // Dims2

    py::implicitly_convertible<std::vector<int>, Dims2>();

    py::class_<DimsHW, Dims2>(m, "DimsHW", DimsHWDoc::descr)
        .def(py::init<>())
        .def(py::init<int, int>(), "h"_a, "w"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dimshw_vector_constructor), "shape"_a)
        // Expose these functions as attributes in Python.
        .def_property("h", [](const DimsHW& dims) { return dims.h(); }, [](DimsHW& dims, int i) { dims.h() = i; })
        .def_property(
            "w", [](const DimsHW& dims) { return dims.w(); }, [](DimsHW& dims, int i) { dims.w() = i; }); // DimsHW

    py::implicitly_convertible<std::vector<int>, DimsHW>();

    // 3D
    py::class_<Dims3, Dims>(m, "Dims3", Dims3Doc::descr)
        .def(py::init<>())
        .def(py::init<int, int, int>(), "dim0"_a, "dim1"_a, "dim2"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dims3_vector_constructor), "shape"_a); // Dims3

    py::implicitly_convertible<std::vector<int>, Dims3>();

    // 4D
    py::class_<Dims4, Dims>(m, "Dims4", Dims4Doc::descr)
        .def(py::init<>())
        .def(py::init<int, int, int, int>(), "dim0"_a, "dim1"_a, "dim2"_a, "dim3"_a)
        // Allows for construction from a tuple/list.
        .def(py::init(lambdas::dims4_vector_constructor), "shape"_a); // Dims4

    py::implicitly_convertible<std::vector<int>, Dims4>();

    py::class_<IHostMemory>(m, "IHostMemory", py::buffer_protocol(), IHostMemoryDoc::descr)
        .def_property_readonly("dtype", [](const IHostMemory& mem) { return mem.type(); })
        .def_property_readonly("nbytes", [](const IHostMemory& mem) { return mem.size(); })
        // Expose buffer interface.
        .def_buffer(lambdas::host_memory_buffer_interface)
        .def("__del__", &utils::doNothingDel<IHostMemory>); // IHostMemory
}

} // namespace tensorrt
