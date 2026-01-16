#include "bayesian-filters/VariationalFreeEnergyFilter.h"

namespace bayesian_filters {
VariarionalFreeEnergyFilter(const NonlinearDynModel& model,
                            const Eigen::MatrixXd& init_state,
                            const Eigen::MatrixXd& process_noise_covariance,
                            const Eigen::MatrixXd& measurement_noise_covariance)
    : model_(model),
      state_(init_state),
      process_noise_covariance_(process_noise_covariance),
      measurement_noise_covariance_(measurement_noise_covariance) {
  // Get the dimension of the state
  int state_dim = model_.getStateDimension();
  int input_dim = model_.getInputDimension();
  int output_dim = model_.getOutputDimension();

  // Check dimensions of initial state and covariance
  if (init_state.rows() != model_.getStateDimension() ||
      init_state.cols() != 1) {
    throw std::invalid_argument("Initial state has incorrect dimensions.");
  }
  // Define VFE function
  casadi::SX x_next = casadi::SX::sym("x_next", state_dim, 1);
  casadi::SX x = casadi::SX::sym("x", state_dim, 1);
  casadi::SX u = casadi::SX::sym("u", input_dim, 1);
  casadi::SX y = casadi::SX::sym("y", output_dim, 1);
  casadi::SX P_x = casadi::SX::sym("P_x", state_dim, state_dim);
  casadi::SX P_y = casadi::SX::sym("P_y", output_dim, output_dim);
  casadi::SX vfe = casadi::SX::sym("vfe", 1, 1);

  // Jacobian of state transition w.r.t. state
  state_transition_jacobian_wrt_state_ =
      jacobian(state_transition_symbolic, state_symbolic_);

  casadi::Function f = model_.getStateTransitionFunction();
  casadi::Function g = model_.getMeasurementFunction();
  vfe = 0.5 * (x_next_prior - f(x, u)).T @P_x @(x_next_prior - f(x, u)) +
        0.5 * (y - g(x)).T @P_y @(y - g(x));

  variational_free_energy_fun_ = casadi::Function("f", {x, u, y}, {vfe});
};

const Eigen::MatrixXd& VariarionalFreeEnergyFilter::predict(
    const Eigen::MatrixXd& input) {
  // Predict the next state
  state_ = model_.stepStateTransition(input, state_);

  // Compute the Jacobian of the state transition function w.r.t. state
  Eigen::MatrixXd y = model_.predictOutput(state_);

  return y;
}

const Eigen::MatrixXd& update(const Eigen::MatrixXd& measurement) {
  // Gradient descent wrt the state
  ;
  // If learning, do gradient descent on model parameters
}

const Eigen::MatrixXd& compute_free_energy(const Eigen::MatrixXd& input,
                                           const Eigen::MatrixXd& measurement) {
  // Get
  return model_.fromCasadiDMtoEigen(
      variational_free_energy_(state_, input, measurement));
}
}  // namespace bayesian_filters
