#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of binding functions
void bind_matrix(py::module_& m);
void bind_vector(py::module_& m);
void bind_activation(py::module_& m);
void bind_loss(py::module_& m);
void bind_neural_network(py::module_& m);
void bind_optimizer(py::module_& m);

PYBIND11_MODULE(fullscratch, m) {
    m.doc() = "FullScratchML: A comprehensive C++ machine learning library";

    // Bind modules
    bind_matrix(m);
    bind_vector(m);
    bind_activation(m);
    bind_loss(m);
    bind_neural_network(m);
    bind_optimizer(m);

    // Version info
    m.attr("__version__") = "0.1.0";
}
