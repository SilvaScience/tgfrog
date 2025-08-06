"""Implementations of scipy optimizations methods."""
from typing import Any, AnyStr, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from tgfrog.loss.mse import mse_loss
from tgfrog.signal_processing.frog_simulator import get_spectrogram_from_parameters


def objective_function(
    x: np.array,
    target_spectrogram: np.ndarray,
    fixed_parameters: Dict[AnyStr, Any],
) -> float:
    """This is a wrapper around the MSE loss function that takes the parameters to fit as a 1D array input.

    This is the format required by scipy optimize. The fixed parameters should contain everything else required to
    estimate a spectrogram.

    Args:
        x: learned coefficients. The last 3 values are the chirp, the other elements are taken to be the phase
            coefficients.
        target_spectrogram: target spectrogram to estimate
        fixed_parameters: dictionary with the parameters that should not be adjusted

    Returns:
        spectrogram_error: MSE between the estimated and target spectrogram
    """
    # add the parameters to optimized to the fixed ones to build an estimated spectrogram
    chirps = x[-3:]
    phase_coefficients = x[:-3]
    all_parameters = fixed_parameters.copy()
    all_parameters["chirps"] = chirps
    all_parameters["phase_coefficients"] = phase_coefficients

    estimated_spectrogram, _ = get_spectrogram_from_parameters(all_parameters)

    spectrogram_error = mse_loss(estimated_spectrogram, target_spectrogram)
    return spectrogram_error


def find_optimal_parameters(
    target_spectrogram: np.ndarray,
    fixed_parameters: Dict[AnyStr, Any],
    num_phi_parameters: int,
    phi_range: Tuple[float, float],
    chirps_range: Tuple[float, float],
    optimizer_method: str = 'Nelder-Mead'
) -> Tuple[np.array, np.array]:
    # create a random initial guess for the phi parameters
    assert phi_range[0] < phi_range[1], f"Invalid range of parameters for the phi parameters. Got {phi_range}."
    assert num_phi_parameters > 0, "At least 1 parameter for phi should be allowed."
    initial_guess_for_phi_parameters = np.random.uniform(
        low=phi_range[0],
        high=phi_range[1],
        size=(num_phi_parameters,)
    )

    assert chirps_range[0] < chirps_range[1], ("Invalid range of parameters for the chirps parameters."
                                               f"Got {chirps_range}.")
    initial_guess_for_chirps = np.random.uniform(
        low=chirps_range[0],
        high=chirps_range[1],
        size=(3,)
    )

    initial_guess_for_parameters = np.concatenate((initial_guess_for_phi_parameters, initial_guess_for_chirps))
    optimized_results = minimize(
        objective_function,
        x0=initial_guess_for_parameters,
        args=(target_spectrogram, fixed_parameters,),
        method=optimizer_method
    )
    assert optimized_results.success, "Optimization failed. Try with a different random seed."
    estimated_parameters = optimized_results.x
    # split in phi parameters and chirps
    return estimated_parameters[:-3], estimated_parameters[-3:]
