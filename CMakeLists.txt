cmake_minimum_required(VERSION 4.0)

project(nn-from-scratch)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED True)


file(GLOB NN_SOURCES CONFIGURE_DEPENDS
     "*.cpp")

add_executable(nn_run ${NN_SOURCES})

target_include_directories(nn_run PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external_submodules/eigen
    ${CMAKE_SOURCE_DIR}/external_submodules/EigenRand
    ${CMAKE_SOURCE_DIR}/external_submodules/mnist/include
)

string(REPLACE "\\" "/" NN_MNIST_ABS "${CMAKE_SOURCE_DIR}/external_submodules/mnist")
target_compile_definitions(nn_run PRIVATE NN_MNIST_DIR=\"${NN_MNIST_ABS}\")