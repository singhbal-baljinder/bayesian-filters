#include <bayesian-filters/KalmanFilter.h>
#include <bayesian-filters/LTIModel.h>

#include <fstream>
#include <iostream>
#include <random>

int main() {
  // We define a simple mass-spring-damper system as our LTI model
  // We assume to measure the position only but want an estimate of the velocity
  // as well The mass is moved using a force (control input)
  Eigen::MatrixXd A(2, 2);
  A << 1.0, 0.1, -0.2, 0.9;
  Eigen::MatrixXd B(2, 1);
  B << 0.0, 0.1;
  Eigen::MatrixXd C(1, 2);
  C << 1.0, 0.0;
  Eigen::MatrixXd D(1, 1);
  D << 0.0;

  Eigen::MatrixXd process_noise_cov(2, 2);
  process_noise_cov << 1e-5, 0.0, 0.0, 1e-5;
  Eigen::MatrixXd measurement_noise_cov(1, 1);
  measurement_noise_cov << 1e-2;

  // Define these once at the start of main()
  std::random_device rd;
  std::mt19937 gen(rd());
  // This represents N(0, 1). You multiply by your desired standard deviation.
  std::normal_distribution<double> normal_distribution(0.0, 1.0);

  bayesian_filters::LTIModel real_lti_model =
      bayesian_filters::LTIModel(A, B, C, Eigen::MatrixXd{});

  // The model in the kalman filter (in general not necessarily the same as the
  // real model)
  bayesian_filters::LTIModel kf_lti_model =
      bayesian_filters::LTIModel(A, B, C, Eigen::MatrixXd{});

  // Initial state and covariance of the Kalman Filter
  Eigen::MatrixXd init_state(2, 1);
  init_state << 0.0, 0.0;
  Eigen::MatrixXd init_covariance = Eigen::MatrixXd::Identity(2, 2) * 1.0;
  bayesian_filters::KalmanFilter kf(kf_lti_model,
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

  std::ofstream data_file("kf_results.csv");
  data_file << "time,true_pos,est_pos,est_vel,std_pos\n";
  // Use a sinusoidal input
  for (int i = 0; i < 100; i++) {
    // Control input
    Eigen::MatrixXd u(1, 1);
    u(0, 0) = 0.5 * std::sin(2 * my_pi * i);
    input[i] = u;

    // True output from the model
    Eigen::MatrixXd true_state = real_lti_model.stepStateTransition(
        u, i == 0 ? init_state : state_estimates[i - 1]);
    Eigen::MatrixXd true_output = real_lti_model.predictOutput(
        u, i == 0 ? init_state : state_estimates[i - 1]);
    // Add measurement noise

    Eigen::MatrixXd noise(1, 1);
    noise(0, 0) = normal_distribution(gen) * 0.01;
    measurements[i] = true_output + noise;

    // Estimate state using Kalman Filter
    kf.predict(u);
    kf.update(measurements[i]);
    state_estimates[i] = kf.getState();
    covariance_estimates[i] = kf.getCovariance();

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
  std::cout << "Data saved to kf_results.csv" << std::endl;
  return 0;
}
