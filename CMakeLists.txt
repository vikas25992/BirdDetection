cmake_minimum_required(VERSION 2.8)
project( trail3 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( TrainSVM TrainSVM.cpp )
target_link_libraries( TrainSVM ${OpenCV_LIBS} )

