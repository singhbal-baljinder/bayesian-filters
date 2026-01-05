add_executable(kalman_filter_example kalman_filter_example.cpp)
target_link_libraries(kalman_filter_example ${PROJECT_NAME})

add_executable(ekf_example ekf_example.cpp)
target_link_libraries(ekf_example ${PROJECT_NAME})

