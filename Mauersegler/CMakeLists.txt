cmake_minimum_required(VERSION 2.8)
project( Mauersegler )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Mauersegler Mauersegler.cpp )
target_link_libraries( Mauersegler ${OpenCV_LIBS} )

