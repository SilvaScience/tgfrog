from typing import Any, AnyStr, Dict, Tuple

import numpy as np
from scipy.fftpack import fft, fftshift

from tgfrog.signal_processing.utils import (filter_wavelength_and_spectrogram,
                                            freq_to_wavelength,
                                            normalization_and_sort_spectrogram,
                                            omega_fs_to_hertz)


def compute_spectrogram_from_time_signal(
    signal_t_tau: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a frequency signal as a function of tau to spatial domain (wavelengths).

    Args:
        signal_t_tau: signal as a function of time t and delay :math:`\tau` as complex values.
            Shape: (time step, tau step)

    Returns:
        signal_freq_tau: signal in frequency domain
        intensity: spectrogram (squared norm of signal in frequency domain)
    """
    assert (
        signal_t_tau.ndim == 2
    ), f"Expected a 2D array to compute a spectrogram.Got an array of shape {signal_t_tau.shape}"
    signal_freq_tau = fftshift(fft(signal_t_tau, axis=0), axes=0)
    # note that signal_freq_tau has complex values, so we need to take the abs value before squaring to get real values
    intensity = np.abs(signal_freq_tau) ** 2
    return signal_freq_tau, intensity


def process_spectrogram(
    spectrogram: np.ndarray, omega: np.array, padding: float = 50.0
) -> Tuple[np.ndarray, np.array]:
    """Post-processing of the spectrogram to filter out-of-range wavelengths and normalize the spectrogram.

    Args:
        spectrogram: computed spectrogram as a (n_wavelength, n_tau) array
        omega: angular frequencies in fs^-1 as a (n_wavelength,) array
        padding: additional range of values outside the target spectrum range in nm. Defaults to 50.

    Returns:
        spectrogram: filtered and normalized
        wavelength: filtered
    """
    # compute wavelengths
    freq = omega_fs_to_hertz(omega)  # convert fs^-1 to s^-1
    wavelength = freq_to_wavelength(freq)
    wavelength, spectrogram = filter_wavelength_and_spectrogram(
        wavelength, spectrogram, padding=padding
    )
    wavelength, spectrogram = normalization_and_sort_spectrogram(
        wavelength, spectrogram
    )
    return spectrogram, wavelength


