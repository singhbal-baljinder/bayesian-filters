#include <bayesian-filters/ExtendedKalmanFilter.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Nonlinear system class Eigen -> Casadi and back", "[nl_dyn_model]") {
  bayesian_filters::NonlinearDynModel nl_model;
  Eigen::MatrixXd test_matrix = Eigen::MatrixXd::Random(3, 2);
  casadi::DM casadi_matrix = nl_model.fromEigentoCasadiDM(test_matrix);
  Eigen::MatrixXd converted_back = nl_model.fromCasadiDMtoEigen(casadi_matrix);
  REQUIRE(test_matrix.isApprox(converted_back, 1e-9));
}

TEST_CASE("Nonlinear system class Casadi -> Eigen and back", "[nl_dyn_model]") {
  bayesian_filters::NonlinearDynModel nl_model;
  casadi::DM test_matrix = casadi::DM::rand(4, 3);
  Eigen::MatrixXd eigen_matrix = nl_model.fromCasadiDMtoEigen(test_matrix);
  casadi::DM converted_back = nl_model.fromEigentoCasadiDM(eigen_matrix);

  REQUIRE((casadi::DM::norm_fro(test_matrix - converted_back) < 1e-9).scalar());
}

TEST_CASE("Nonlinear system class state transition", "[nl_dyn_model]") {
  // Define a simple nonlinear state transition: x_next = x^2 + u
  casadi::SX state_symbolic = casadi::SX::sym("x", 2, 1);
  casadi::SX input_symbolic = casadi::SX::sym("u", 1, 1);
  casadi::SX state_transition_symbolic =
      casadi::SX::zeros(2, 1);
  state_transition_symbolic(0) = state_symbolic(0) * state_symbolic(0) + input_symbolic(0);
  state_transition_symbolic(1) = state_symbolic(1) * state_symbolic(1) + input_symbolic(0);

  casadi::SX measurement_symbolic = casadi::SX::zeros(1, 1);
  measurement_symbolic(0) = state_symbolic(0) + state_symbolic(1);

  bayesian_filters::NonlinearDynModel nl_model(
      state_transition_symbolic,
      measurement_symbolic,
      state_symbolic,
      input_symbolic);

  Eigen::MatrixXd test_state(2, 1);
  test_state << 2.0, 3.0;
  Eigen::MatrixXd test_input(1, 1);
  test_input << 1.0;

  Eigen::MatrixXd next_state = nl_model.stepStateTransition(test_input, test_state);

  Eigen::MatrixXd expected_next_state(2, 1);
  expected_next_state << 5.0, 10.0;  // [2^2 + 1; 3^2 + 1]

  REQUIRE(next_state.isApprox(expected_next_state, 1e-9));
}
