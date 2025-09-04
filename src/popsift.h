#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <popsift/popsift.h>
#include <popsift/features.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_config.h>
#include <popsift/version.hpp>
#include <popsift/common/device_prop.h>

#include <vector>

namespace py = pybind11;

namespace pps{

typedef py::array_t<float, py::array::c_style | py::array::forcecast> pyarray_f;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> pyarray_d;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> pyarray_int;
typedef py::array_t<unsigned char, py::array::c_style | py::array::forcecast> pyarray_uint8;

template <typename T>
py::array_t<T> py_array_from_data(const T *data, size_t shape0) {
  py::array_t<T> res({shape0});
  std::copy(data, data + shape0, res.mutable_data());
  return res;
}

template <typename T>
py::array_t<T> py_array_from_data(const T *data, size_t shape0, size_t shape1) {
  py::array_t<T> res({shape0, shape1});
  std::copy(data, data + shape0 * shape1, res.mutable_data());
  return res;
}

template <typename T>
py::array_t<T> py_array_from_data(const T *data, size_t shape0, size_t shape1, size_t shape2) {
  py::array_t<T> res({shape0, shape1, shape2});
  std::copy(data, data + shape0 * shape1 * shape2, res.mutable_data());
  return res;
}

class PopSiftContext{
    PopSift *ps;
    popsift::Config *config;

    float peak_threshold = 0;
    float edge_threshold = 0;
    bool use_root = true;
    float downsampling = 0;
public:
    PopSiftContext();
    ~PopSiftContext();

    void setup(float peak_threshold,
                float edge_threshold,
                bool use_root,
                float downsampling);

    PopSift *get();
};

py::object popsift(pyarray_uint8 image,
                 float peak_threshold,
                 float edge_threshold,
                 int target_num_features,
                 bool use_root,
                 float downsampling);

bool fitsTexture(int width, int height, float downsampling);

}
