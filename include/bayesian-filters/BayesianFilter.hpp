

#pragma once

namespace bayesian_filters {

template <typename StateType, typename MeasurementType>
class BayesianFilter {
 public:
  virtual ~BayesianFilter() = default;

  virtual const StateType& update(const MeasurementType& measurement) = 0;
};
}  // namespace bayesian_filters
