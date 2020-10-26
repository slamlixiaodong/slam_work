cmake_minimum_required(VERSION 2.8)
project(gaussnewton)
find_package(OpenCV REQUIRED)
include_directories(
        "/usr/include/eigen3"
        ${OpenCV_INCLUDE_DIRS}
)
add_executable(gaussnewton gaussnewton.cpp)
target_link_libraries(gaussnewton ${OpenCV_LIBS})