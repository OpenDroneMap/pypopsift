#include <pybind11/pybind11.h>
#include "popsift.h"

namespace py = pybind11;

PYBIND11_MODULE(pypopsift, m) {
    m.doc() = R"pbdoc(
        pypopsift: Python module for CUDA accelerated GPU SIFT 
        -----------------------

        .. currentmodule:: pypopsift

        .. autosummary::
           :toctree: _generate

           popsift
    )pbdoc";

    m.def("popsift", pps::popsift,
        py::arg("image"),
        py::arg("peak_threshold") = 0.1,
        py::arg("edge_threshold") = 10,
        py::arg("target_num_features") = 4000,
        py::arg("use_root") = true,
        py::arg("downsampling") = -1
    );

    m.def("fits_texture", pps::fitsTexture,
            py::arg("width"),
            py::arg("height"),
            py::arg("downsampling") = -1);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
