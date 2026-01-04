#pragma once
#include "bayesian-filters/BayesianFilter.hpp"
#include "bayesian-filters/LTIModel.h"

namespace bayesian_filters {

class KalmanFilter : public BayesianFilter<Eigen::MatrixXd, Eigen::MatrixXd> {
 public:
  KalmanFilter() = default;
  KalmanFilter(
      const LTIModel& model,
      const Eigen::MatrixXd& init_state = Eigen::MatrixXd{},
      const Eigen::MatrixXd& init_covariance = Eigen::MatrixXd{},
      const Eigen::MatrixXd& process_noise_covariance = Eigen::MatrixXd{},
      const Eigen::MatrixXd& measurement_noise_covariance = Eigen::MatrixXd{});
  ~KalmanFilter() = default;

  // Implements the prediction step of the Kalman Filter
  const Eigen::MatrixXd& predict(
      const Eigen::MatrixXd& input = Eigen::MatrixXd{});

  // Implements the update step of the Kalman Filter
  const Eigen::MatrixXd& update(
      const Eigen::MatrixXd& measurement = Eigen::MatrixXd{}) override;

  const Eigen::MatrixXd& getState();
  const Eigen::MatrixXd& getCovariance();

 private:
  Eigen::MatrixXd state_{};
  Eigen::MatrixXd covariance_{};
  LTIModel model_;
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
};

}  // namespace bayesian_filters
