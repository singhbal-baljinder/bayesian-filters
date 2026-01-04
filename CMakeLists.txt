cmake_minimum_required(VERSION 3.16)

project(bayesian-filters VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} 
 src/KalmanFilter.cpp
 src/LTIModel.cpp)

target_include_directories(${PROJECT_NAME} 
 PUBLIC 
  # Path for users of the library
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
 PRIVATE
  # Path for internal .cpp files
  ${CMAKE_CURRENT_SOURCE_DIR}/src)


target_link_libraries(${PROJECT_NAME} 
 PUBLIC 
  Eigen3::Eigen)

install(TARGETS ${PROJECT_NAME}
 EXPORT ${PROJECT_NAME}Targets
 LIBRARY DESTINATION lib
 ARCHIVE DESTINATION lib
 RUNTIME DESTINATION bin
)

# Examples
add_executable(kalman_filter_example examples/kalman_filter_example.cpp)
target_link_libraries(kalman_filter_example ${PROJECT_NAME})



