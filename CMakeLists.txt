cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(hashing_algorithms VERSION 0.1.0 LANGUAGES CXX CUDA)

add_executable(hashing_algorithms add.cu)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})