#include <bayesian-filters/ExtendedKalmanFilter.h>
#include <bayesian-filters/NonlinearDynModel.h>

#include <casadi/casadi.hpp>
#include <fstream>
#include <iostream>
#include <random>

int main() {
  // Parameters
  double dt = 0.01;
  double g = 9.81;
  double L = 1.0;
  double b = 0.8;  // Damping

  // 1. Define Nonlinear Pendulum Dynamics in CasADi
  casadi::SX x = casadi::SX::sym("x", 2, 1);  // [theta; theta_dot]
  casadi::SX u = casadi::SX::sym("u", 1, 1);  // torque

  casadi::SX theta = x(0);
  casadi::SX theta_dot = x(1);

  // x_next = x + f(x,u)*dt
  casadi::SX theta_next = theta + theta_dot * dt;
  casadi::SX theta_dot_next =
      theta_dot + (-(g / L) * casadi::SX::sin(theta) - b * theta_dot + u) * dt;

  casadi::SX state_transition_symbolic =
      casadi::SX::vertcat({theta_next, theta_dot_next});
  casadi::SX measurement_symbolic = theta;  // We only measure position (angle)

  // 2. Setup Model and EKF
  bayesian_filters::NonlinearDynModel pendulum_model(
      state_transition_symbolic, measurement_symbolic, x, u);

  Eigen::MatrixXd process_noise_cov(2, 2);
  process_noise_cov << 1e-4, 0.0, 0.0, 1e-4;
  Eigen::MatrixXd measurement_noise_cov(1, 1);
  measurement_noise_cov << 1e-2;

  Eigen::MatrixXd init_state(2, 1);
  init_state << 0.5, 0.0;  // Start at 0.5 rad
  Eigen::MatrixXd init_covariance = Eigen::MatrixXd::Identity(2, 2) * 0.1;

  bayesian_filters::ExtendedKalmanFilter ekf(pendulum_model,
                                             init_state,
                                             init_covariance,
                                             process_noise_cov,
                                             measurement_noise_cov);

  // 3. Simulation Loop
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 1.0);

  std::ofstream data_file("ekf_results.csv");
  data_file << "time,true_pos,est_pos,est_vel,std_pos\n";

  Eigen::MatrixXd x_true = init_state;

  for (int i = 0; i < 1000; i++) {
    Eigen::MatrixXd u_in(1, 1);
    u_in(0, 0) = 0;  // Zero torque

    // Update Ground Truth
    x_true = pendulum_model.stepStateTransition(u_in, x_true);
    Eigen::MatrixXd y_true = pendulum_model.predictOutput(x_true);

    // Create Measurement (Truth + Noise)
    Eigen::MatrixXd z = y_true;
    z(0, 0) += dist(gen) * std::sqrt(measurement_noise_cov(0, 0));

    // EKF Step
    ekf.predict(u_in);
    ekf.update(z);

    // Save
    data_file << i * dt << "," << z(0, 0) << "," << ekf.getState()(0, 0) << ","
              << ekf.getState()(1, 0) << ","
              << std::sqrt(ekf.getCovariance()(0, 0)) << "\n";
  }

  data_file.close();
  std::cout << "Pendulum EKF simulation complete. Data saved." << std::endl;
  return 0;
}
