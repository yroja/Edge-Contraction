cmake_minimum_required(VERSION 3.15)
project(greedy_joining_spanning_forest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)
find_package(pybind11 REQUIRED)


pybind11_add_module(greedy_joining_spanning_forest python-interface.cxx)
install(TARGETS greedy_joining_spanning_forest DESTINATION .)