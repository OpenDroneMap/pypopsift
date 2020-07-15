cmake_minimum_required(VERSION 3.14)
project(pypopsift)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

set(PopSift_BUILD_EXAMPLES OFF)
FetchContent_Declare(
  popsift
  GIT_REPOSITORY https://github.com/alicevision/popsift/
  GIT_TAG        b584a125be8154c81a2b6cd46cc73e2b3a014e38
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(popsift)

add_subdirectory(pybind11)

pybind11_add_module(pypopsift src/main.cpp src/popsift.cpp)
target_link_libraries(pypopsift PUBLIC popsift)