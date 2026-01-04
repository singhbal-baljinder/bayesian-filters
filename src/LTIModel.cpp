
#include "bayesian-filters/LTIModel.h"

#include <iostream>
#include <stdexcept>
namespace bayesian_filters {

LTIModel::LTIModel(const Eigen::MatrixXd& A,
                   const Eigen::MatrixXd& B,
                   const Eigen::MatrixXd& C,
                   const Eigen::MatrixXd& D) {
  // Sanity checks. At minimum, A and C must be provided
  if (A.rows() == 0 || A.cols() == 0) {
    throw std::invalid_argument("Matrix A must not be empty.");
  }

  if (C.rows() == 0 || C.cols() == 0) {
    throw std::invalid_argument("Matrix C must not be empty.");
  }

  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Matrix A must be square.");
  }

  if (C.cols() != A.rows()) {
    throw std::invalid_argument(
        "Matrix C must have the same number of columns as the number of rows "
        "in A.");
  }
  // Set A and C
  A_ = A;
  C_ = C;

  // D is optional. If provided, check its dimensions. Otherwise, set to zero
  if (D.rows() > 0) {
    if ((D.rows() != C.rows() || D.cols() != B.cols())) {
      throw std::invalid_argument(
          "Matrix D must have the same number of rows as C and the same number "
          "of columns as B.");
    }

    D_ = D;

  } else {
    D_ = Eigen::MatrixXd::Zero(C.rows(), B.cols());
  }

  // B is optional. If provided, check its dimensions. Otherwise, set to zero
  if (B.rows() > 0) {
    if (B.rows() != A.rows()) {
      throw std::invalid_argument(
          "Matrix B must have the same number of rows as A.");
    }

    B_ = B;

  } else {
    B_ = Eigen::MatrixXd::Zero(A.rows(), 1);
  }
}

const Eigen::MatrixXd LTIModel::stepStateTransition(
    const Eigen::MatrixXd& input, const Eigen::MatrixXd& curr_state) {
  Eigen::MatrixXd next_state;
  // If no input is provided, assume zero input
  if (input.rows() == 0 || input.cols() == 0) {
    next_state = A_ * curr_state;
    return next_state;
  }

  // Check that input dimensions are compatible
  if (input.rows() != B_.cols()) {
    throw std::invalid_argument(
        "Input matrix has incorrect dimensions for state transition.");
  }

  next_state = A_ * curr_state + B_ * input;
  return next_state;
}

const Eigen::MatrixXd LTIModel::predictOutput(
    const Eigen::MatrixXd& input, const Eigen::MatrixXd& curr_state) {
  Eigen::MatrixXd output;
  if (input.rows() == 0 || input.cols() == 0) {
    output = C_ * curr_state;
    return output;
  } else {
    // Check that input dimensions are compatible
    if (input.rows() != D_.cols()) {
      throw std::invalid_argument(
          "Input matrix has incorrect dimensions for output prediction.");
    }
  }

  output = C_ * curr_state + D_ * input;

  return output;
}

void LTIModel::setA(const Eigen::MatrixXd& A) { A_ = A; }
void LTIModel::setB(const Eigen::MatrixXd& B) { B_ = B; }
void LTIModel::setC(const Eigen::MatrixXd& C) { C_ = C; }
void LTIModel::setD(const Eigen::MatrixXd& D) { D_ = D; }

const Eigen::MatrixXd& LTIModel::getA() { return A_; }
const Eigen::MatrixXd& LTIModel::getB() { return B_; }
const Eigen::MatrixXd& LTIModel::getC() { return C_; }
const Eigen::MatrixXd& LTIModel::getD() { return D_; }
}  // namespace bayesian_filters
