#include "bayesian-filters/VariationalFreeEnergyFilter.h"

namespace bayesian_filters {

VariationalFreeEnergyFilter::VariationalFreeEnergyFilter(
    const NonlinearDynModel& model,
    const Eigen::MatrixXd& init_state,
    const Eigen::MatrixXd& process_noise_covariance,
    const Eigen::MatrixXd& measurement_noise_covariance,
    const std::map<std::string, int>& opts)
    : model_(model),
      process_noise_covariance_(process_noise_covariance),
      measurement_noise_covariance_(measurement_noise_covariance) {
  // Get the dimension of the state
  int state_dim = model_.getStateDimension();
  int input_dim = model_.getInputDimension();
  int output_dim = model_.getOutputDimension();

  // Check dimensions of initial state and covariance
  if (init_state.rows() != state_dim || init_state.cols() != 1) {
    throw std::invalid_argument("Initial state has incorrect dimensions.");
  }
  state_ = init_state;
  // Define the symbolic variables for the VFE function definition
  casadi::SX x_next = casadi::SX::sym("x_next", state_dim, 1);
  casadi::SX x = casadi::SX::sym("x", state_dim, 1);
  casadi::SX u = casadi::SX::sym("u", input_dim, 1);
  casadi::SX y = casadi::SX::sym("y", output_dim, 1);
  // Use implicit casadi convertion from DM to SX
  casadi::SX P_x = model_.fromEigentoCasadiDM(process_noise_covariance);
  casadi::SX P_y = model_.fromEigentoCasadiDM(measurement_noise_covariance);
  casadi::SX vfe = casadi::SX::sym("vfe", 1, 1);

  casadi::Function f = model_.getStateTransitionFunction();
  casadi::Function g = model_.getMeasurementFunction();

  std::vector<casadi::SX> args_f = {u, x};
  // NOTA BENE: Casadi returns std::vector<casadi::SX>>, so we need to take the
  // first element
  casadi::SX error_x = x_next - f(args_f).at(0);
  casadi::SX tmp_var_x = mtimes(error_x.T(), P_x);

  std::vector<casadi::SX> args_g = {x};
  // NOTA BENE: Casadi returns std::vector<casadi::SX>>, so we need to take the
  // first element
  casadi::SX error_y = y - g(args_g).at(0);
  casadi::SX tmp_var_y = mtimes(error_y.T(), P_y);

  // VFE definition
  vfe = 0.5 * mtimes(tmp_var_x, error_x) + 0.5 * mtimes(tmp_var_y, error_y);

  vfe_fun_ = casadi::Function("f", {u, x, y, x_next}, {vfe});

  // Precompute gradients wrt state
  casadi::SX dvfe_dx = jacobian(vfe, x);
  casadi::SX dvfe_du = jacobian(vfe, u);
  dvfe_dx_fun_ = casadi::Function("dvfe_dx", {u, x, y, x_next}, {dvfe_dx});

  // Parse options
  learn_itr_ = opts.at("N");
  learn_rate_ = opts.at("alpha");
};

std::vector<Eigen::MatrixXd> VariationalFreeEnergyFilter::predict(
    const Eigen::MatrixXd& input) {
  // Save input
  last_input_ = input;
  // Store current state before update
  last_x_prior_ = state_;
  // Predict the next state
  state_ = model_.stepStateTransition(input, state_);
  last_x_next_prior_ = state_;  // Store predicted state
  // Predict output
  Eigen::MatrixXd y = model_.predictOutput(state_);
  std::vector<Eigen::MatrixXd> res(2);
  res[0] = state_;
  res[1] = y;
  return res;
}

const Eigen::MatrixXd& VariationalFreeEnergyFilter::update(
    const Eigen::MatrixXd& measurement) {
  // Preperate input for jacobian function
  casadi::DM u_cs = model_.fromEigentoCasadiDM(last_input_);
  casadi::DM x_prior_cs = model_.fromEigentoCasadiDM(last_x_prior_);
  casadi::DM y_cs = model_.fromEigentoCasadiDM(measurement);
  casadi::DM x_next_prior_cs = model_.fromEigentoCasadiDM(last_x_next_prior_);

  std::vector<casadi::DM> args_dvfe_dx = {
      u_cs, x_prior_cs, y_cs, x_next_prior_cs};
  // Gradient descent wrt the state
  for (int i = 0; i < learn_itr_; ++i) {
    state_ = state_ - learn_rate_ * model_.fromCasadiDMtoEigen(
                                        dvfe_dx_fun_(args_dvfe_dx)[0].T());
  }
  // If learning, do gradient descent on model parameters

  return state_;
}

const Eigen::MatrixXd VariationalFreeEnergyFilter::compute_free_energy(
    const Eigen::MatrixXd& input,
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXd& measurement,
    const Eigen::MatrixXd& x_next) {
  // Prepare inputs for casadi function
  casadi::DM u_cs = model_.fromEigentoCasadiDM(input);
  casadi::DM x_cs = model_.fromEigentoCasadiDM(x);
  casadi::DM y_cs = model_.fromEigentoCasadiDM(measurement);
  casadi::DM x_next_cs = model_.fromEigentoCasadiDM(x_next);

  // Use casadi function to compute VFE
  std::vector<casadi::DM> args_dvfe_dx = {u_cs, x_cs, y_cs, x_next_cs};
  Eigen::MatrixXd vfe = model_.fromCasadiDMtoEigen(vfe_fun_(args_dvfe_dx)[0]);

  return vfe;
}

Eigen::MatrixXd VariationalFreeEnergyFilter::getState() { return state_; }
NonlinearDynModel VariationalFreeEnergyFilter::getModel() { return model_; }
}  // namespace bayesian_filters
