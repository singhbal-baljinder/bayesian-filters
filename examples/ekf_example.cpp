#include <bayesian-filters/ExtendedKalmanFilter.h>
#include <bayesian-filters/KalmanFilter.h>
#include <bayesian-filters/LTIModel.h>
#include <bayesian-filters/NonlinearDynModel.h>

#include <casadi/casadi.hpp>
#include <fstream>
#include <iostream>
#include <random>

int main() {
  // We define a simple mass-spring-damper system as our LTI model
  // We assume to measure the position only but want an estimate of the velocity
  // as well The mass is moved using a force (control input)
  std::vector<double> A_data(4, 0);
  A_data[0] = 1.0;
  A_data[1] = 0.1;
  A_data[2] = -0.2;
  A_data[3] = 0.9;
  std::vector<double> B_data(2, 0);
  B_data[0] = 0.0;
  B_data[1] = 0.1;
  std::vector<double> C_data(2, 0);
  C_data[0] = 1.0;
  C_data[1] = 0.0;
  std::vector<double> D_data(1, 0);
  D_data[0] = 0.0;
  Eigen::MatrixXd A =
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::ColMajor>>(A_data.data());
  Eigen::MatrixXd B =
      Eigen::Map<Eigen::Matrix<double, 2, 1, Eigen::ColMajor>>(B_data.data());
  Eigen::MatrixXd C =
      Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>>(C_data.data());
  Eigen::MatrixXd D =
      Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>>(D_data.data());

  Eigen::MatrixXd process_noise_cov(2, 2);
  process_noise_cov << 1e-5, 0.0, 0.0, 1e-5;
  Eigen::MatrixXd measurement_noise_cov(1, 1);
  measurement_noise_cov << 1e-2;

  // Define these once at the start of main()
  std::random_device rd;
  std::mt19937 gen(rd());
  // This represents N(0, 1). You multiply by your desired standard deviation.
  std::normal_distribution<double> normal_distribution(0.0, 1.0);

  bayesian_filters::LTIModel lti_model =
      bayesian_filters::LTIModel(A, B, C, Eigen::MatrixXd{});

  // The model in the kalman filter (in general not necessarily the same as the
  // real model)
  casadi::SX state_symbolic = casadi::SX::sym("x", 2, 1);
  casadi::SX input_symbolic = casadi::SX::sym("u", 1, 1);

  // Set the symbolic matrives to the numerical values
  casadi::SX A_symbolic = casadi::SX::zeros(2, 2);
  A_symbolic(0, 0) = 1.0;
  A_symbolic(0, 1) = 0.1;
  A_symbolic(1, 0) = -0.2;
  A_symbolic(1, 1) = 0.9;
  casadi::SX B_symbolic = casadi::SX::zeros(2, 1);
  B_symbolic(0, 0) = 0.0;
  B_symbolic(1, 0) = 0.1;
  casadi::SX C_symbolic = casadi::SX::zeros(1, 2);
  C_symbolic(0, 0) = 1.0;

  casadi::SX state_transition_symbolic =
      mtimes(A_symbolic, state_symbolic) + mtimes(B_symbolic, input_symbolic);
  casadi::SX measurement_symbolic = mtimes(C_symbolic, state_symbolic);
  bayesian_filters::NonlinearDynModel nonlinear_model =
      bayesian_filters::NonlinearDynModel(state_transition_symbolic,
                                          measurement_symbolic,
                                          state_symbolic,
                                          input_symbolic);

  // Initial state and covariance of the Kalman Filter
  Eigen::MatrixXd init_state(2, 1);
  init_state << 0.0, 0.0;
  Eigen::MatrixXd init_covariance = Eigen::MatrixXd::Identity(2, 2) * 1.0;
  bayesian_filters::ExtendedKalmanFilter ekf(nonlinear_model,
                                             init_state,
                                             init_covariance,
                                             process_noise_cov,
                                             measurement_noise_cov);

  // Simulate some measurements (for example purposes, we use the true model
  // output plus noise)
  int num_steps = 100;
  std::vector<Eigen::MatrixXd> measurements(num_steps);
  std::vector<Eigen::MatrixXd> input(num_steps);
  std::vector<Eigen::MatrixXd> state_estimates(num_steps);
  std::vector<Eigen::MatrixXd> covariance_estimates(num_steps);

  float period = 10.0;
  double my_pi = 3.14159265358979323846;
  // Angular vel of a sinusoidal input [rad/iteration]
  float angular_freq = 0.1 * 2 * my_pi / period;

  std::ofstream data_file("ekf_results.csv");
  data_file << "time,true_pos,est_pos,est_vel,std_pos\n";
  // Use a sinusoidal input
  for (int i = 0; i < 100; i++) {
    // Control input
    Eigen::MatrixXd u(1, 1);
    u(0, 0) = 0.5 * std::sin(2 * my_pi * i);
    input[i] = u;

    // True output from the model
    Eigen::MatrixXd true_state = lti_model.stepStateTransition(
        u, i == 0 ? init_state : state_estimates[i - 1]);
    Eigen::MatrixXd true_output = lti_model.predictOutput(
        u, i == 0 ? init_state : state_estimates[i - 1]);
    // Add measurement noise

    Eigen::MatrixXd noise(1, 1);
    noise(0, 0) = normal_distribution(gen) * 0.01;
    measurements[i] = true_output + noise;

    // Estimate state using Kalman Filter
    ekf.predict(u);
    ekf.update(measurements[i]);
    state_estimates[i] = ekf.getState();
    covariance_estimates[i] = ekf.getCovariance();

    // Save in  a file or process the estimates as needed
    double time = i * 0.1;  // Assuming dt = 0.1
    double est_pos = state_estimates[i](0, 0);
    double est_vel = state_estimates[i](1, 0);
    double std_pos =
        std::sqrt(covariance_estimates[i](0, 0));  // Standard deviation

    data_file << time << "," << measurements[i](0, 0)
              << ","  // Measurement (Noisy Pos)
              << est_pos << "," << est_vel << "," << std_pos << "\n";
  }

  data_file.close();
  std::cout << "Data saved to ekf_results.csv" << std::endl;
  return 0;
}
