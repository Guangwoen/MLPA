aux_source_directory(. TEST_LIST)
file(GLOB MAIN_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
include_directories(/usr/local/include ${MAIN_HEAD})
include_directories(/Users/cuiguangyuan/Documents/Programs/libs/C++/dist/include/coin-or)
link_directories(/usr/local/lib)
link_directories(/Users/cuiguangyuan/Documents/Programs/libs/C++/dist/lib)

add_executable(MLPA_TEST ${TEST_LIST})
target_link_libraries(MLPA_TEST Matplot++::matplot Clp OsiClp CoinUtils qpOASES mlpa lapack blas gtest gtest_main)