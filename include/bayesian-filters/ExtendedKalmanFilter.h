#pragma once
#include <eigen3/Eigen/Dense>

#include "bayesian-filters/BayesianFilter.h"
#include "bayesian-filters/NonlinearDynModel.h"
namespace bayesian_filters {

class ExtendedKalmanFilter
    : public BayesianFilter<Eigen::MatrixXd, Eigen::MatrixXd> {
 public:
  ExtendedKalmanFilter() = default;
  ExtendedKalmanFilter(
      const NonlinearDynModel& models,
      const Eigen::MatrixXd& init_state = Eigen::MatrixXd{},
      const Eigen::MatrixXd& init_covariance = Eigen::MatrixXd{},
      const Eigen::MatrixXd& process_noise_covariance = Eigen::MatrixXd{},
      const Eigen::MatrixXd& measurement_noise_covariance = Eigen::MatrixXd{});
  ~ExtendedKalmanFilter() = default;

  // Implements the prediction step of the Extended Kalman Filter
  const Eigen::MatrixXd& predict(
      const Eigen::MatrixXd& input = Eigen::MatrixXd{});

  // Implements the update step of the Extended Kalman Filter
  const Eigen::MatrixXd& update(
      const Eigen::MatrixXd& measurement = Eigen::MatrixXd{}) override;

  const Eigen::MatrixXd& getState();
  const Eigen::MatrixXd& getCovariance();

 private:
  Eigen::MatrixXd state_{};
  Eigen::MatrixXd covariance_{};
  NonlinearDynModel model_;
  // Innovation covariance
  Eigen::MatrixXd S_{};
  // Kalman Gain
  Eigen::MatrixXd K_{};
  // Identity matrix for covariance update
  Eigen::MatrixXd I_{};
  // Process noise covariance
  Eigen::MatrixXd process_noise_covariance_{};
  // Measurement noise covariance
  Eigen::MatrixXd measurement_noise_covariance_{};
  // To ease debugging store state and input dimensions
  int state_dim_{0};
  int input_dim_{0};
};
}  // namespace bayesian_filters
