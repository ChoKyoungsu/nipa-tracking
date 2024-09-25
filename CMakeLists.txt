cmake_minimum_required(VERSION 3.10)
project(my_opencv_project)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV 경로 설정
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)

add_executable(main2 main2.cpp)

# OpenCV 라이브러리 링크
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main2 ${OpenCV_LIBS})

# set(SOURCE_FILES main.c)
# add_executable(main ${SOURCE_FILES})
# target_link_libraries(main ${OpenCV_LIBRARIES})

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
