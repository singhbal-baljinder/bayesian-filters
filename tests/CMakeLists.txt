find_package(Catch2 REQUIRED)
include(Catch) # Usually provided by Catch2 installation
enable_testing()

set(TEST_SOURCES
    NonlinearDynModelTest.cpp
)

add_executable(unit_tests ${TEST_SOURCES})

target_link_libraries(unit_tests 
    PRIVATE 
    bayesian-filters 
    Catch2::Catch2WithMain
    Eigen3::Eigen
    casadi
)

catch_discover_tests(unit_tests)
