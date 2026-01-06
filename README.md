# 🧠 bayesian-filters

A compact C++ library providing implementations of common Bayesian filters
for linear and nonlinear dynamical systems. Currently, the project includes a
Kalman Filter, an Extended Kalman Filter (EKF), simple linear time-invariant
(LTI) models, and a helper `NonlinearDynModel` based on CasADi for symbolic
model definition.

✨ **Currently available**
- `KalmanFilter` (class): prediction and update steps for LTI systems.
	- `predict(input)` — run the prediction step.
	- `update(measurement)` — incorporate a measurement and update state.
	- `getState()` / `getCovariance()` — access current state and covariance.
- `ExtendedKalmanFilter` (class): EKF for nonlinear systems using CasADi for automatic differentiation.
	- `predict(input)` — prediction using the nonlinear state transition.
	- `update(measurement)` — measurement update using linearized Jacobians.
	- `getState()` / `getCovariance()` — access current state and covariance.
- `LTIModel` (class): simple linear state-space model utilities.
	- `stepStateTransition(input, curr_state)`, `predictOutput(input,curr_state)`.
	- setters/getters for `A,B,C,D` matrices.
- `NonlinearDynModel` (class) ✨: CasADi-backed symbolic models for EKF.
	- `stepStateTransition(...)`, `predictOutput(...)` and Jacobian helpers.

⚙️ **Build (Linux / macOS)**
Prerequisites: `cmake`, a C++ compiler, `Eigen3`, `CasADi`, and
Python3 for plotting. You can create the conda environment defined in
`conda_ci_env.yml` (recommended):

```
conda env create -f conda_ci_env.yml -n bayesian-f
conda actiate bayesian-f
```

Example build steps from the repo path:
```
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_INSTALL_PREFIX=./install
cmake --build . --target install 
```

If CMake configured examples and tests, the example binaries will be placed
in `build/examples/`. Run them as:
```
./build/examples/ekf_example 
./build/examples/kalman_filter_example
```

📊 **Run and plot examples**
The KF and EKF examples generate a file `<path-where-the-example-were-laumched>/ekf_results.csv`/ To plot results:
```
# (use the conda env from conda_ci_env.yml if desired)
python3 scripts/plot_results.py <path-where-the-example-were-laumched>/ekf_results.csv
```


✅ **Testing**
After building, run the unit tests with CTest (from the `build/tests` directory):
```
cd build/tests
ctest --output-on-failure
```
