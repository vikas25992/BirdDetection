cmake_minimum_required(VERSION 2.8)
project( Birds )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Birds Birds.cpp )
target_link_libraries( Birds ${OpenCV_LIBS} )

