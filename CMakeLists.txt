cmake_minimum_required(VERSION 3.16)

project(bayesian-filters VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)

add_library(${PROJECT_NAME} 
 src/KalmanFilter.cpp
 src/LTIModel.cpp
 src/NonlinearDynModel.cpp
 src/ExtendedKalmanFilter.cpp)

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
  casadi::casadi
  Eigen3::Eigen)

install(TARGETS ${PROJECT_NAME}
 EXPORT ${PROJECT_NAME}Targets
 LIBRARY DESTINATION lib
 ARCHIVE DESTINATION lib
 RUNTIME DESTINATION bin
)

set(BUILD_EXAMPLES ON CACHE BOOL "Build the examples for this library")
# Examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
set(BUILD_TESTING ON CACHE BOOL "Build the unit tests for this library")
# Tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
