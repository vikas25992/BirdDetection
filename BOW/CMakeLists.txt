cmake_minimum_required(VERSION 2.8)
project( BOW )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( BOW BOW.cpp )
target_link_libraries( BOW ${OpenCV_LIBS} )

