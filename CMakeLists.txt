cmake_minimum_required( VERSION 3.0 )
project( useLK )

set( CMAKE_BUILD_TYPE Debug )

find_package(Eigen3 REQUIRED)
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${EIGEN3_INCLUDE_DIRS} )
include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

add_executable( useLK main.cpp )
target_link_libraries( useLK klt_lib)
