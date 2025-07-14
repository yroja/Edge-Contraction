#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <greedy_joining_spanning_forest.hxx>  // Deine Header-Datei

namespace py = pybind11;

// Wrap die Funktion als Python-Modul
PYBIND11_MODULE(greedy_joining_spanning_forest, m) {
    m.def("greedy_joining_extended", &greedy_joining_extended<double>);
    m.def("greedy_joining", &greedy_joining<double>);
}