cmake_minimum_required(VERSION 2.8)
project(disparity)

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(OpenCV 3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${Pangolin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3"
)
add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES} )