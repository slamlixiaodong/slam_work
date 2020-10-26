cmake_minimum_required(VERSION 2.8)
project(GN-BA)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
find_package(Sophus REQUIRED)
include_directories(
        "/usr/include/eigen3"
        ${Sophus_INCLUDE_DIRECTORIES}
)
add_executable(GN-BA GN-BA.cpp)
target_link_libraries(GN-BA ${Sophus_LIBRARIES})