cmake_minimum_required(VERSION 3.4)
project(sayhelloslam)
#包含引用目录
include_directories(
${PROJECT_SOURCE_DIR}/include
)
#make install的设置
INSTALL(FILES include/hello.h DESTINATION /usr/local/include)
INSTALL(FILES build/libhello.so DESTINATION /usr/local/lib)
#设置可执行文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
#生成库文件
add_library(hello SHARED src/hello.cpp)
#生成可执行文件
add_executable(sayhello src/useHello.cpp)
#链接库
target_link_libraries(sayhello hello)


