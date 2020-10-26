cmake_minimum_required(VERSION 3.5)
project(usegeometry)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories("/usr/include/eigen3")
set(SOURCE_FILES main.cpp)
add_executable(usegeometry ${SOURCE_FILES})
