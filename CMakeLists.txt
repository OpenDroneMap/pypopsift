cmake_minimum_required(VERSION 3.14)
project(pypopsift)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

set(PopSift_BUILD_EXAMPLES OFF)
FetchContent_Declare(
  popsift
  GIT_REPOSITORY https://github.com/uav4geo/popsift/
  GIT_TAG        develop
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(popsift)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        aa982e131d723e2e428245847c0bc75693a11569
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(depname)

  add_subdirectory(${pybind11_SOURCE_DIR})
endif()

add_subdirectory(pybind11)

pybind11_add_module(pypopsift src/main.cpp src/popsift.cpp)
target_link_libraries(pypopsift PUBLIC popsift)