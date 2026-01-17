#pragma once
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>

#include "bayesian-filters/BayesianFilter.h"
#include "bayesian-filters/NonlinearDynModel.h"

namespace bayesian_filters {

class VariationalFreeEnergyFilter
    : public BayesianFilter<Eigen::MatrixXd, Eigen::MatrixXd> {
 public:
  VariationalFreeEnergyFilter() = default;

  VariationalFreeEnergyFilter(
      const NonlinearDynModel& model,
      const Eigen::MatrixXd& init_state = Eigen::MatrixXd{},
      const Eigen::MatrixXd& process_noise_covariance = Eigen::MatrixXd{},
      const Eigen::MatrixXd& measurement_noise_covariance = Eigen::MatrixXd{},
      const std::map<std::string, int>& opts = {{"N", 1}, {"alpha", 0.001}});

  ~VariationalFreeEnergyFilter() = default;

  // Implements a forward pass using the current model
  std::vector<Eigen::MatrixXd> predict(
      const Eigen::MatrixXd& input = Eigen::MatrixXd{});

  // Implements an inference step
  const Eigen::MatrixXd& update(
      const Eigen::MatrixXd& measurement = Eigen::MatrixXd{}) override;

  // Compute Free Energy using Laplace approximation
  const Eigen::MatrixXd compute_free_energy(const Eigen::MatrixXd& input,
                                            const Eigen::MatrixXd& x,
                                            const Eigen::MatrixXd& measurement,
                                            const Eigen::MatrixXd& x_next);

  Eigen::MatrixXd getState();
  NonlinearDynModel getModel();

 protected:
  NonlinearDynModel model_;
  Eigen::MatrixXd state_{};
  Eigen::MatrixXd last_input_{};
  Eigen::MatrixXd last_x_prior_{};
  Eigen::MatrixXd last_x_next_prior_{};

  int learn_itr_{1};
  float learn_rate_{0.001};

  // Process noise covariance
  Eigen::MatrixXd process_noise_covariance_{};
  // Measurement noise covariance
  Eigen::MatrixXd measurement_noise_covariance_{};
  // Symbolic Variational Free Energy function
  casadi::Function vfe_fun_{};
  // Symbolic function representing the gradient of the Variational Free Energy
  // wrt the state
  casadi::Function dvfe_dx_fun_{};
  // Symbolic function representing the gradient of the Variational Free Energy
  // wrt the input
  casadi::Function dvfe_du_fun_{};
};

}  // namespace bayesian_filters
