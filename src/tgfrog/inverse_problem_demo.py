"""Main entry point for the POC.

This generates a FROG simulation using a set of parameters. The resulting spectrogram is then passed to a solver than
estimates these parameters.
"""
import time
from typing import Any, AnyStr, Dict

import numpy as np

from tgfrog.plotting_tools.plot_frog_trace import plot_estimated_and_target_spectrogram
from tgfrog.signal_processing.frog_simulator import get_spectrogram_from_parameters
from tgfrog.solver.scipy_optimize import find_optimal_parameters


SEED = 42
RANDOM_PARAMETERS = True
NUM_PHASE_COEFFICIENTS = 4
PARAMETERS_SEED = 42
RUN_NAME = "run4"


def get_parameters(randomize_values: bool = False, seed: int = 0, num_phase_coefficients: int = 3
                   ) -> Dict[AnyStr, Any]:
    """Get the parameters for the FROG simulation.

    This could be replaced by an external yaml file and wrapped in a dataclass.

    Returns:
        parameters in a dictionary
    """
    if randomize_values:
        np.random.seed(seed)

    n_time_step = 2048  # number of time steps
    time_delta = 1.0  # time step dt in fs
    lambda_zero = 800  # central wavelength in nm
    delta_lambda = 40  # spectral width in nm
    chirps = [200, 200, 200]  # should be fitted
    phase_coefficients = [0, 0, 500]  # can be arbitrary length in rad, fs, fs^2, fs^3... - should be fitted
    if randomize_values:
        chirps = np.random.uniform(low=-200, high=200, size=(3,)).tolist()
        phase_coefficients = np.random.uniform(low=-500, high=500, size=(num_phase_coefficients,)).tolist()

    tau_range = [-200, 200]
    n_tau = 400

    assert tau_range[1] > tau_range[0], ("Time delay tau should span a linear space i.e. stop range should be larger"
                                         f"than start range. Got {tau_range}")

    tau_values = np.linspace(tau_range[0], tau_range[1], n_tau)

    assert time_delta > 0, f"time resolution should be positive. Got {time_delta}"

    return dict(
        n_time_step=n_time_step,
        time_delta=time_delta,
        lambda_zero=lambda_zero,
        delta_lambda=delta_lambda,
        chirps=chirps,
        phase_coefficients=phase_coefficients,
        tau_values=tau_values,
    )


def main():
    # this could be moved to a "make 1 spectrogram randomly and solve it" method
    # the main() would call this several times to get a better evaluation of a solver
    # parameters could be randomized in a range instead of fixed

    parameters = get_parameters(
        randomize_values=RANDOM_PARAMETERS,
        num_phase_coefficients=NUM_PHASE_COEFFICIENTS,
        seed=PARAMETERS_SEED
    )
    np.random.seed(SEED)

    target_spectrogram, wavelength = get_spectrogram_from_parameters(parameters)
    target_phase_coefficients = parameters["phase_coefficients"]
    target_chirps = parameters["chirps"]
    fixed_parameters = parameters.copy()
    del fixed_parameters["chirps"]
    del fixed_parameters["phase_coefficients"]

    start_time = time.time()

    calculated_phi_parameters, calculated_chirps = find_optimal_parameters(
        target_spectrogram,
        fixed_parameters,
        num_phi_parameters=len(target_phase_coefficients),
        phi_range=[-600, 600],
        chirps_range=[-300, 300],
        optimizer_method="Nelder-Mead"
    )
    best_parameters = fixed_parameters.copy()
    best_parameters["phase_coefficients"] = calculated_phi_parameters
    best_parameters["chirps"] = calculated_chirps
    estimated_spectrogram, _ = get_spectrogram_from_parameters(best_parameters)

    print("Phi Parameters")
    target_phase_coefficients = np.concatenate((target_phase_coefficients, np.zeros((2,))))
    for i, (estimated_phi, true_phi) in enumerate(zip(calculated_phi_parameters, target_phase_coefficients)):
        print(f"Order {i}, true: {true_phi:0.2f}   - estimated: {estimated_phi:0.2f}")
    print("\nChirps Parameters")
    for i, (estimated_c, true_c) in enumerate(zip(calculated_chirps, target_chirps)):
        print(f"Parameter {i + 1}, true: {true_c:0.2f}   - estimated: {estimated_c:0.2f}")

    plot_estimated_and_target_spectrogram(
        estimated_spectrogram,
        target_spectrogram,
        tau=parameters["tau_values"],
        wavelength=wavelength,
        spectrum_padding=50,
        output_path=f"demo_{RUN_NAME}.png"
    )
    print(f"\nSolver run time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
