#include "bayesian-filters/ExtendedKalmanFilter.h"

namespace bayesian_filters {

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const NonlinearDynModel& model,
    const Eigen::MatrixXd& init_state,
    const Eigen::MatrixXd& init_covariance,
    const Eigen::MatrixXd& process_noise_covariance,
    const Eigen::MatrixXd& measurement_noise_covariance)
    : model_(model),
      state_(init_state),
      covariance_(init_covariance),
      process_noise_covariance_(process_noise_covariance),
      measurement_noise_covariance_(measurement_noise_covariance) {
  // Check dimensions of initial state and covariance
  if (init_state.rows() != model_.getStateDimension() ||
      init_state.cols() != 1) {
    throw std::invalid_argument("Initial state has incorrect dimensions.");
  }

  if (init_covariance.rows() != model_.getStateDimension() ||
      init_covariance.cols() != model_.getStateDimension()) {
    throw std::invalid_argument("Initial covariance has incorrect dimensions.");
  }

  // Initialize matrices with appropriate dimensions
  state_ = init_state;
  covariance_ = init_covariance;
  I_ = Eigen::MatrixXd::Identity(model_.getStateDimension(),
                                 model_.getStateDimension());
}

const Eigen::MatrixXd& ExtendedKalmanFilter::predict(
    const Eigen::MatrixXd& input) {
  // Predict the next state
  state_ = model_.stepStateTransition(input, state_);

  // Compute the Jacobian of the state transition function w.r.t. state
  Eigen::MatrixXd F =
      model_.computeStateTransitionJacobianWrtState(input, state_);

  // Predict covariance matrix
  covariance_ = F * covariance_ * F.transpose() + process_noise_covariance_;
  return state_;
}

const Eigen::MatrixXd& ExtendedKalmanFilter::update(
    const Eigen::MatrixXd& measurement) {
  // Compute the Jacobian of the measurement function w.r.t. state
  Eigen::MatrixXd H = model_.computeMeasurementJacobianWrtState(state_);

  // Innovation of the covariance
  S_ = H * covariance_ * H.transpose() + measurement_noise_covariance_;

  // Kalman Gain
  K_ = covariance_ * H.transpose() * S_.inverse();

  // Update state and covariance
  state_ = state_ + K_ * (measurement - model_.predictOutput(state_));
  covariance_ = (I_ - K_ * H) * covariance_;

  return state_;
}

const Eigen::MatrixXd& ExtendedKalmanFilter::getState() { return state_; }
const Eigen::MatrixXd& ExtendedKalmanFilter::getCovariance() {
  return covariance_;
}
}  // namespace bayesian_filters
