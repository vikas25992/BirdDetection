cmake_minimum_required(VERSION 2.8)
project( Back_ground )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Back_ground Back_ground.cpp )
target_link_libraries( Back_ground ${OpenCV_LIBS} )

