cmake_minimum_required(VERSION 2.8)
project(trajectory)

set( CMAKE_BUILD_TYPE "release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${Pangolin_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3"
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(trajectory draw_trajectory.cpp)
target_link_libraries( trajectory ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES} )
