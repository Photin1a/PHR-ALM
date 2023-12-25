cmake_minimum_required(VERSION 3.10)
project(alm)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Start to deal with the project: ${PROJECT_NAME}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(alm test.cpp)
# add_executable(vir vir.cpp)