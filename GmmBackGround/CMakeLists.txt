# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(GmmBG)    #!
set(CMAKE_BUILD_TYPE "Debug")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(
    OpenCV REQUIRED
)

include_directories(. 
    ${OpenCV_INCLUDE_DIRS} 
    ./include
    /opt/MVS/include
)

link_directories(
  /opt/MVS/lib/64
  /opt/MVS/lib/32
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

aux_source_directory(src PROJ_SRC)

add_executable(GmmBG src/main.cpp src/CameraCtl.cpp)


target_link_libraries(GmmBG
    MvCameraControl
    libMvCameraControl.so
    ${OpenCV_LIBS}
)
