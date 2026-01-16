#pragma once
#include <casadi/casadi.hpp>
#include <eigen/Eigen/Dense.hpp>

namespace bayesian_filters {

class VariationalFreeEnergyFilter
    : public BayesianFilter<Eigen::MatrixXd, Eigen::MatrixXd> {
 public:
  VariationalFreeEnergyFilter() = default;

  VariationalFreeEnergyFilter(
      const NonlinearDynModel& model,
      const Eigen::MatrixXd& init_state = Eigen::MatrixXd{},
      const Eigen::MatrixXd& process_noise_covariance = Eigen::MatrixXd{},
      const Eigen::MatrixXd& measurement_noise_covariance = Eigen::MatrixXd{});

  ~VariationalFreeEnergyFilter() = default;

  // Implements a forward pass using the current model
  const Eigen::MatrixXd& predict(
      const Eigen::MatrixXd& input = Eigen::MatrixXd{});

  // Implements an inference step
  const Eigen::MatrixXd& update(
      const Eigen::MatrixXd& measurement = Eigen::MatrixXd{}) override;

  // Compute Free Energy using Laplace approximation
  const Eigen::MatrixXd& compute_free_energy(const Eigen::MatrixXd& input, const Eigen::MatrixXd& measurement);

 protected:
  Eigen::MatrixXd state_;
  Eigen::MatrixXd model_;
  // Process noise covariance
  Eigen::MatrixXd process_noise_covariance_{};
  // Measurement noise covariance
  Eigen::MatrixXd measurement_noise_covariance_{};
  // Symbolic Variational Free Energy function
  casadi::Funtion variational_free_energy_fun_{};
}

}  // namespace bayesian_filters
