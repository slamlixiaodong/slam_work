cmake_minimum_required(VERSION 3.6)
project(useEigen)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(
        "/usr/include/eigen3"
)
set(SOURCE_FILES main.cpp)
add_executable(useEigen ${SOURCE_FILES})