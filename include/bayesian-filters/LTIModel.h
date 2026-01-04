#pragma once
#include <Eigen/Dense>

namespace bayesian_filters {
class LTIModel {
 public:
  LTIModel(const Eigen::MatrixXd& A = Eigen::MatrixXd{},
           const Eigen::MatrixXd& B = Eigen::MatrixXd{},
           const Eigen::MatrixXd& C = Eigen::MatrixXd{},
           const Eigen::MatrixXd& D = Eigen::MatrixXd{});
  ~LTIModel() = default;

  // State transition step based on current state and input
  const Eigen::MatrixXd stepStateTransition(
      const Eigen::MatrixXd& input = Eigen::MatrixXd{},
      const Eigen::MatrixXd& curr_state = Eigen::MatrixXd{});

  // Predict output based on current state and input
  const Eigen::MatrixXd predictOutput(
      const Eigen::MatrixXd& input = Eigen::MatrixXd{},
      const Eigen::MatrixXd& curr_state = Eigen::MatrixXd{});

  // Set methods
  void setA(const Eigen::MatrixXd& A);
  void setB(const Eigen::MatrixXd& B);
  void setC(const Eigen::MatrixXd& C);
  void setD(const Eigen::MatrixXd& D);

  // Get methods
  const Eigen::MatrixXd& getA();
  const Eigen::MatrixXd& getB();
  const Eigen::MatrixXd& getC();
  const Eigen::MatrixXd& getD();

 private:
  // State transition matrices
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
  // Output matrices
  Eigen::MatrixXd C_;
  Eigen::MatrixXd D_;
};
}  // namespace bayesian_filters
