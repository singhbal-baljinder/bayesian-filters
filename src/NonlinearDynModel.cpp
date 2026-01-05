#include "bayesian-filters/NonlinearDynModel.h"

namespace bayesian_filters {
NonlinearDynModel::NonlinearDynModel(
    const casadi::SX& state_transition_symbolic,
    const casadi::SX& measurement_symbolic,
    const casadi::SX& state_symbolic,
    const casadi::SX& input_symbolic)
    : state_symbolic_(state_symbolic), input_symbolic_(input_symbolic) {
  // Make sure the state and input are column vectors

  if (state_symbolic.columns() != 1) {
    throw std::invalid_argument(
        "State symbolic variable must be a column vector.");
  }
  if (input_symbolic.columns() != 1) {
    throw std::invalid_argument(
        "Input symbolic variable must be a column vector.");
  }
  state_transition_function_ = casadi::Function(
      "f", {input_symbolic_, state_symbolic_}, {state_transition_symbolic});
  measurement_function_ =
      casadi::Function("h", {state_symbolic_}, {measurement_symbolic});

  // Jacobian of state transition w.r.t. state
  state_transition_jacobian_wrt_state_ =
      jacobian(state_transition_symbolic, state_symbolic_);

  state_transition_jacobian_wrt_state_fun_ =
      casadi::Function("dfdx",
                       {input_symbolic_, state_symbolic_},
                       {state_transition_jacobian_wrt_state_});
  // Jacobian of state transition w.r.t. input
  state_transition_jacobian_wrt_input_ =
      jacobian(state_transition_symbolic, input_symbolic_);

  state_transition_jacobian_wrt_input_fun_ =
      casadi::Function("dfdx",
                       {input_symbolic_, state_symbolic_},
                       {state_transition_jacobian_wrt_input_});
  // Jacobian of measurement w.r.t. state
  measurement_jacobian_wrt_state_ =
      jacobian(measurement_symbolic, state_symbolic_);

  measurement_jacobian_wrt_state_fun_ = casadi::Function(
      "dfdx", {state_symbolic_}, {measurement_jacobian_wrt_state_});
  // Store state and input dimensions
  state_dim_ = state_symbolic_.rows();
  input_dim_ = input_symbolic_.rows();
}

Eigen::MatrixXd NonlinearDynModel::stepStateTransition(
    const Eigen::MatrixXd& input, const Eigen::MatrixXd& current_state) {
  casadi::DM current_state_casadi = this->fromEigentoCasadiDM(current_state);
  casadi::DM input_casadi = this->fromEigentoCasadiDM(input);

  std::vector<casadi::DM> args = {input_casadi, current_state_casadi};
  std::vector<casadi::DM> next_state_casadi = state_transition_function_(args);

  return this->fromCasadiDMtoEigen(next_state_casadi[0]);
}

Eigen::MatrixXd NonlinearDynModel::predictOutput(
    const Eigen::MatrixXd& current_state) {
  casadi::DM current_state_casadi = this->fromEigentoCasadiDM(current_state);
  casadi::DM measurement_casadi = measurement_function_(current_state_casadi);

  return this->fromCasadiDMtoEigen(measurement_casadi);
}

casadi::DM NonlinearDynModel::fromEigentoCasadiDM(
    const Eigen::MatrixXd& eigen_matrix) {
  std::vector<double> data_vector(eigen_matrix.data(),
                                  eigen_matrix.data() + eigen_matrix.size());
  return casadi::DM(
      casadi::Sparsity::dense(eigen_matrix.rows(), eigen_matrix.cols()),
      data_vector);
}

Eigen::Map<const Eigen::MatrixXd> NonlinearDynModel::fromCasadiDMtoEigen(
    const casadi::DM& casadi_matrix) {
  return Eigen::Map<const Eigen::MatrixXd>(
      casadi_matrix.ptr(), casadi_matrix.size1(), casadi_matrix.size2());
}

Eigen::MatrixXd NonlinearDynModel::computeStateTransitionJacobianWrtState(
    const Eigen::MatrixXd& input, const Eigen::MatrixXd& current_state) {
  casadi::DM current_state_casadi = this->fromEigentoCasadiDM(current_state);
  casadi::DM input_casadi = this->fromEigentoCasadiDM(input);

  std::vector<casadi::DM> args = {input_casadi, current_state_casadi};

  return this->fromCasadiDMtoEigen(
      state_transition_jacobian_wrt_state_fun_(args)[0]);
}

Eigen::MatrixXd NonlinearDynModel::computeStateTransitionJacobianWrtInput(
    const Eigen::MatrixXd& input, const Eigen::MatrixXd& current_state) {
  casadi::DM current_state_casadi = this->fromEigentoCasadiDM(current_state);
  casadi::DM input_casadi = this->fromEigentoCasadiDM(input);

  std::vector<casadi::DM> args = {input_casadi, current_state_casadi};
  return this->fromCasadiDMtoEigen(
      state_transition_jacobian_wrt_input_fun_(args)[0]);
}

Eigen::MatrixXd NonlinearDynModel::computeMeasurementJacobianWrtState(
    const Eigen::MatrixXd& current_state) {
  casadi::DM current_state_casadi = this->fromEigentoCasadiDM(current_state);

  std::vector<casadi::DM> args = {current_state_casadi};
  return this->fromCasadiDMtoEigen(
      measurement_jacobian_wrt_state_fun_(args)[0]);
}

const int& NonlinearDynModel::getStateDimension() { return state_dim_; }
const int& NonlinearDynModel::getInputDimension() { return input_dim_; }

}  // namespace bayesian_filters
