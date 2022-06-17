###############################################################################
## project settings ###########################################################
###############################################################################
# CMake documentation: https://cmake.org/cmake/help/v3.2/
# CMake simple example: http://derekmolloy.ie/hello-world-introductions-to-cmake/
# CMake cuda example: https://codeyarns.com/2013/09/13/how-to-build-cuda-programs-using-cmake/

# this project is set up such that it compiles
project(cudasamples LANGUAGES CUDA CXX)

# compiler options, set c++11 (requires cmake 3.2 or higher)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11) 

###############################################################################
## find libraries #############################################################
###############################################################################
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# FIND LIBRARIES
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

###############################################################################
## file globbing ##############################################################
###############################################################################

# find source and header files.
set(main_ "main_cuda.cu")

###############################################################################
## target definitions #########################################################
###############################################################################

# COMPILER FLAGS AND DEFINITIONS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}) # ;-gencode arch=compute_30,code=sm_30)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

# SPECIFY TARGETS
cuda_add_executable(main ${main_})

target_link_libraries(main ${OpenCV_LIBS})

###############################################################################
## testing ####################################################################
###############################################################################



###############################################################################
## output #####################################################################
###############################################################################
