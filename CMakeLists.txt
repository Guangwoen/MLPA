cmake_minimum_required(VERSION 3.29)
project(MLPA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

find_package(Matplot++ REQUIRED)

include_directories(/usr/local/include)
file(GLOB MAIN_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
include_directories(${MAIN_HEAD})
include_directories(/Users/cuiguangyuan/Documents/Programs/libs/C++/dist/include/coin-or)
link_directories(/usr/local/lib)
link_directories(/Users/cuiguangyuan/Documents/Programs/libs/C++/dist/lib)

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(MLPA main.cpp ${MAIN_SRC})

target_link_libraries(MLPA Matplot++::matplot Clp OsiClp CoinUtils qpOASES lapack blas)

add_library(mlpa ${MAIN_SRC})

add_subdirectory(test)
