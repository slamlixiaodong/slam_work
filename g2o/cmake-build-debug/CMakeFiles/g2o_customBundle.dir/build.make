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
CMAKE_SOURCE_DIR = /home/lixiaodong/project/g2o

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lixiaodong/project/g2o/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/g2o_customBundle.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/g2o_customBundle.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/g2o_customBundle.dir/flags.make

CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o: CMakeFiles/g2o_customBundle.dir/flags.make
CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o: ../g2o_bundle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lixiaodong/project/g2o/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o -c /home/lixiaodong/project/g2o/g2o_bundle.cpp

CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lixiaodong/project/g2o/g2o_bundle.cpp > CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.i

CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lixiaodong/project/g2o/g2o_bundle.cpp -o CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.s

CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.requires:

.PHONY : CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.requires

CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.provides: CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.requires
	$(MAKE) -f CMakeFiles/g2o_customBundle.dir/build.make CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.provides.build
.PHONY : CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.provides

CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.provides.build: CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o


# Object files for target g2o_customBundle
g2o_customBundle_OBJECTS = \
"CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o"

# External object files for target g2o_customBundle
g2o_customBundle_EXTERNAL_OBJECTS =

g2o_customBundle: CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o
g2o_customBundle: CMakeFiles/g2o_customBundle.dir/build.make
g2o_customBundle: /home/lixiaodong/Sophus/build/libSophus.so
g2o_customBundle: /usr/lib/x86_64-linux-gnu/libcholmod.so
g2o_customBundle: /usr/lib/x86_64-linux-gnu/libamd.so
g2o_customBundle: /usr/lib/x86_64-linux-gnu/libcolamd.so
g2o_customBundle: /usr/lib/x86_64-linux-gnu/libcamd.so
g2o_customBundle: /usr/lib/x86_64-linux-gnu/libccolamd.so
g2o_customBundle: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
g2o_customBundle: libBALProblem.so
g2o_customBundle: libParseCmd.so
g2o_customBundle: CMakeFiles/g2o_customBundle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lixiaodong/project/g2o/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable g2o_customBundle"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/g2o_customBundle.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/g2o_customBundle.dir/build: g2o_customBundle

.PHONY : CMakeFiles/g2o_customBundle.dir/build

CMakeFiles/g2o_customBundle.dir/requires: CMakeFiles/g2o_customBundle.dir/g2o_bundle.cpp.o.requires

.PHONY : CMakeFiles/g2o_customBundle.dir/requires

CMakeFiles/g2o_customBundle.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/g2o_customBundle.dir/cmake_clean.cmake
.PHONY : CMakeFiles/g2o_customBundle.dir/clean

CMakeFiles/g2o_customBundle.dir/depend:
	cd /home/lixiaodong/project/g2o/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lixiaodong/project/g2o /home/lixiaodong/project/g2o /home/lixiaodong/project/g2o/cmake-build-debug /home/lixiaodong/project/g2o/cmake-build-debug /home/lixiaodong/project/g2o/cmake-build-debug/CMakeFiles/g2o_customBundle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/g2o_customBundle.dir/depend

