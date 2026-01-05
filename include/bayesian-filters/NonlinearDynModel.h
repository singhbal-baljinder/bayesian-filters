#pragma once
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>

namespace bayesian_filters {
class NonlinearDynModel {
 public:
  NonlinearDynModel() = default;
  NonlinearDynModel(const casadi::SX& state_transition_symbolic,
                    const casadi::SX& measurement_symbolic,
                    const casadi::SX& state_symbolic,
                    const casadi::SX& input_symbolic);
  ~NonlinearDynModel() = default;

  /**
   *  @brief Computes the state derivative given the current state and input,
   * using the state transition function.
   *  @param current_state The current state of the system
   *  @param input The control input to the system
   *  @return The state derivative as a casadi::SX object
   */
  Eigen::MatrixXd stepStateTransition(const Eigen::MatrixXd& input,
                                      const Eigen::MatrixXd& current_state);

  /**
   * @brief Computes the measurement given the current state using the
   * measurement function.
   * @param current_state The current state of the system
   * @return The measurement as a casadi::SX object
   */
  Eigen::MatrixXd predictOutput(const Eigen::MatrixXd& current_state);

  /**
   * @brief Computes the Jacobian of the state transition function
   * with respect to the state at the given current state and input.
   * @param current_state The current state of the system
   * @param input The control input to the system
   * @return The state Jacobian as an Eigen::MatrixXd
   */
  Eigen::MatrixXd computeStateTransitionJacobianWrtState(
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& current_state);

  /**
   * @brief Computes the Jacobian of the state transition function
   * with respect to the input at the given current state and input.
   * @param current_state The current state of the system
   * @param input The control input to the system
   * @return The input Jacobian as an Eigen::MatrixXd
   */
  Eigen::MatrixXd computeStateTransitionJacobianWrtInput(
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& current_state);

  /**
   * @brief Computes the Jacobian of the measurement function
   * with respect to the state at the given current state.
   * @param current_state The current state of the system
   * @return The measurement Jacobian as an Eigen::MatrixXd
   */
  Eigen::MatrixXd computeMeasurementJacobianWrtState(
      const Eigen::MatrixXd& current_state);

  /**
   * @brief Converts an Eigen matrix to a CasADi DM matrix.
   * @param eigen_matrix The input Eigen matrix.
   * @return The corresponding CasADi DM matrix.
   */
  casadi::DM fromEigentoCasadiDM(const Eigen::MatrixXd& eigen_matrix);

  /**
   * @brief Converts a CasADi DM matrix to an Eigen matrix.
   * @param casadi_matrix The input CasADi DM matrix.
   * @return The corresponding Eigen matrix.
   */
  Eigen::Map<const Eigen::MatrixXd> fromCasadiDMtoEigen(
      const casadi::DM& casadi_matrix);

  const int& getStateDimension();
  const int& getInputDimension();

 private:
  casadi::Function state_transition_function_{};
  casadi::Function measurement_function_{};
  casadi::SX state_symbolic_{};
  casadi::SX input_symbolic_{};
  casadi::SX state_transition_jacobian_wrt_state_{};
  casadi::SX state_transition_jacobian_wrt_input_{};
  casadi::SX measurement_jacobian_wrt_state_{};
  casadi::Function state_transition_jacobian_wrt_state_fun_{};
  casadi::Function state_transition_jacobian_wrt_input_fun_{};
  casadi::Function measurement_jacobian_wrt_state_fun_{};
  // To ease debugging store state and input dimensions
  int state_dim_{0};
  int input_dim_{0};
};
}  // namespace bayesian_filters
