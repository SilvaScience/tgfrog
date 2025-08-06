import numpy as np


def mse_loss(estimated_spectrogram: np.ndarray, target_spectrogram: np.ndarray) -> float:
    """Squared error averaged over pixels for an estimated spectrogram given the real one.

    Args:
        estimated_spectrogram: estimated spectrogram as a 2D array [n_tau, n_wavelength]
        target_spectrogram: real spectrogram to estimate as a 2D array [n_tau, n_wavelength]

    Returns:
        mean squared error
    """
    return np.mean((estimated_spectrogram - target_spectrogram) ** 2)
