from typing import Tuple

import numpy as np

from tgfrog.utils.constants import (METER_TO_NANOMETER, SECOND_TO_FEMTOSECOND,
                                    SPECTRUM_RANGE, SPEED_OF_LIGHT, TWOPI)


def omega_fs_to_hertz(omega: np.ndarray) -> np.ndarray:
    r"""Convert a signal frequency :math:`\omega` in 1 / fs to a frequency in Hz.

    Args:
        omega: array of frequencies in :math:`fs^{-1}`

    Returns:
        frequencies in Hz
    """
    freq = omega * SECOND_TO_FEMTOSECOND  # convert to Hz
    return freq


def freq_to_wavelength(freq: np.array) -> np.array:
    """Convert frequencies in Hz to wavelength in nm.

    Args:
        freq: array of frequencies in Hz.

    Returns:
        wavelengths in nm
    """
    nonzero_mask = freq != 0
    wavelength = np.zeros_like(freq)
    wavelength[nonzero_mask] = (
        TWOPI * SPEED_OF_LIGHT * METER_TO_NANOMETER / freq[nonzero_mask]
    )

    return wavelength


def filter_wavelength_and_spectrogram(
    wavelength: np.array, spectrogram: np.ndarray, padding: float
) -> Tuple[np.array, np.ndarray]:
    """Filter out wavelengths outside the target range.

    Args:
        wavelength: wavelength array (in nm)
        spectrogram: spectrogram as a (n_wavelength, n_tau) array
        padding: wavelength padding outside the spectrum range

    Returns:
        wavelength: values in the spectrum range
        spectrogram: values in the spectrum range
    """
    assert (
        wavelength.shape[0] == spectrogram.shape[0]
    ), "wavelength array size do not match spectrogram"

    # keep only wavelengths in the target spectrum range within
    valid_mask = (wavelength >= SPECTRUM_RANGE[0] - padding) & (
        wavelength <= SPECTRUM_RANGE[1] + padding
    )
    # filtering
    wavelength = wavelength[valid_mask]
    intensity = spectrogram[valid_mask]

    # check that some wavelengths remain in the range
    assert len(wavelength) > 0, "No values are in the specified range."

    return wavelength, intensity


def normalization_and_sort_spectrogram(
    wavelength: np.array, spectrogram: np.ndarray
) -> Tuple[np.array, np.ndarray]:
    """Normalize and sort wavelength and spectrogram.

    Args:
        wavelength: wavelength array (in nm)
        spectrogram: spectrogram as a (n_wavelength, n_tau) array

    Returns:
        wavelength: values in the spectrum range
        spectrogram: values in the spectrum range
    """
    # Normalization
    spectrogram = spectrogram / max(np.max(spectrogram), 1e-10)
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    spectrogram = spectrogram[sort_idx, :]

    return wavelength, spectrogram
