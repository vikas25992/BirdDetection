cmake_minimum_required(VERSION 2.8)
project( test_hist )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( test_hist test_hist.cpp )
target_link_libraries( test_hist ${OpenCV_LIBS} )

