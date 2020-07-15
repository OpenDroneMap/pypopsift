#include <pybind11/pybind11.h>
#include "popsift.h"

namespace py = pybind11; // TODO REMOVE?

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
        py::arg("target_num_features") = 4000
    );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
