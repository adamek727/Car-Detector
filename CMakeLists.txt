cmake_minimum_required(VERSION 3.21)
project(Car_Detection)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(Car_Detection src/main.cpp)
target_link_libraries(Car_Detection ${OpenCV_LIBS} )
