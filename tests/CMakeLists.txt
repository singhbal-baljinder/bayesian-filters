set(TEST_SOURCES
    test_nl_dyn_model.cpp
)

add_executable(run_unit_tests ${TEST_SOURCES})

target_link_libraries(run_unit_tests 
    PRIVATE 
    bayesian-filters 
    Catch2::Catch2WithMain  # This provides the entry point for the tests
    Eigen3::Eigen
    casadi
)
