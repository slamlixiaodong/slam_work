cmake_minimum_required(VERSION 3.0)
project(direct_method)
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
find_package(OpenCV 3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
include_directories(
        ${Pangolin_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/Eigen3"
)
add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES} )