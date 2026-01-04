#include "bayesian-filters/KalmanFilter.h"

namespace bayesian_filters {

KalmanFilter::KalmanFilter(const LTIModel& model,
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
  if (init_state.rows() != model_.getA().rows() || init_state.cols() != 1) {
    throw std::invalid_argument("Initial state has incorrect dimensions.");
  }

  if (init_covariance.rows() != model_.getA().rows() ||
      init_covariance.cols() != model_.getA().rows()) {
    throw std::invalid_argument("Initial covariance has incorrect dimensions.");
  }

  // Initialize matrices with appropriate dimensions
  state_ = init_state;
  covariance_ = init_covariance;
  I_ = Eigen::MatrixXd::Identity(model_.getA().rows(), model_.getA().rows());
}

const Eigen::MatrixXd& KalmanFilter::predict(const Eigen::MatrixXd& input) {
  // Predict the next state
  state_ = model_.stepStateTransition(input, state_);

  // Predict convariance matrix
  covariance_ = model_.getA() * covariance_ * model_.getA().transpose() +
                process_noise_covariance_;
  return state_;
}

const Eigen::MatrixXd& KalmanFilter::update(
    const Eigen::MatrixXd& measurement) {
  // Innovation of the covariance
  S_ = model_.getC() * covariance_ * model_.getC().transpose() +
       measurement_noise_covariance_;

  // Kalman Gain
  K_ = covariance_ * model_.getC().transpose() * S_.inverse();

  // Update state and covariance
  state_ = state_ +
           K_ * (measurement - model_.predictOutput(Eigen::MatrixXd{}, state_));
  covariance_ = (I_ - K_ * model_.getC()) * covariance_;

  return state_;
}

const Eigen::MatrixXd& KalmanFilter::getState() { return state_; }
const Eigen::MatrixXd& KalmanFilter::getCovariance() { return covariance_; }
}  // namespace bayesian_filters
