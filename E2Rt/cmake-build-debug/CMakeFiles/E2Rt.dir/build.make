# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/lixiaodong/clion-2018.1.5/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/lixiaodong/clion-2018.1.5/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lixiaodong/project/E2Rt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lixiaodong/project/E2Rt/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/E2Rt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/E2Rt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/E2Rt.dir/flags.make

CMakeFiles/E2Rt.dir/E2Rt.cpp.o: CMakeFiles/E2Rt.dir/flags.make
CMakeFiles/E2Rt.dir/E2Rt.cpp.o: ../E2Rt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lixiaodong/project/E2Rt/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/E2Rt.dir/E2Rt.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/E2Rt.dir/E2Rt.cpp.o -c /home/lixiaodong/project/E2Rt/E2Rt.cpp

CMakeFiles/E2Rt.dir/E2Rt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/E2Rt.dir/E2Rt.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lixiaodong/project/E2Rt/E2Rt.cpp > CMakeFiles/E2Rt.dir/E2Rt.cpp.i

CMakeFiles/E2Rt.dir/E2Rt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/E2Rt.dir/E2Rt.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lixiaodong/project/E2Rt/E2Rt.cpp -o CMakeFiles/E2Rt.dir/E2Rt.cpp.s

CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires:

.PHONY : CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires

CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides: CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires
	$(MAKE) -f CMakeFiles/E2Rt.dir/build.make CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides.build
.PHONY : CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides

CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides.build: CMakeFiles/E2Rt.dir/E2Rt.cpp.o


# Object files for target E2Rt
E2Rt_OBJECTS = \
"CMakeFiles/E2Rt.dir/E2Rt.cpp.o"

# External object files for target E2Rt
E2Rt_EXTERNAL_OBJECTS =

E2Rt: CMakeFiles/E2Rt.dir/E2Rt.cpp.o
E2Rt: CMakeFiles/E2Rt.dir/build.make
E2Rt: /home/lixiaodong/Sophus/build/libSophus.so
E2Rt: CMakeFiles/E2Rt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lixiaodong/project/E2Rt/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable E2Rt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/E2Rt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/E2Rt.dir/build: E2Rt

.PHONY : CMakeFiles/E2Rt.dir/build

CMakeFiles/E2Rt.dir/requires: CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires

.PHONY : CMakeFiles/E2Rt.dir/requires

CMakeFiles/E2Rt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/E2Rt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/E2Rt.dir/clean

CMakeFiles/E2Rt.dir/depend:
	cd /home/lixiaodong/project/E2Rt/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lixiaodong/project/E2Rt /home/lixiaodong/project/E2Rt /home/lixiaodong/project/E2Rt/cmake-build-debug /home/lixiaodong/project/E2Rt/cmake-build-debug /home/lixiaodong/project/E2Rt/cmake-build-debug/CMakeFiles/E2Rt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/E2Rt.dir/depend

