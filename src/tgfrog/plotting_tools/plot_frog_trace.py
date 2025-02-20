from sys import intern
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from tgfrog.plotting_tools import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from tgfrog.signal_processing.utils import (filter_wavelength_and_spectrogram,
                                            freq_to_wavelength,
                                            normalization_and_sort_spectrogram,
                                            omega_fs_to_hertz)
from tgfrog.utils.constants import SPECTRUM_RANGE

plt.style.use(PLOT_STYLE_PATH)


def plot_frog_trace(
    spectrogram: np.ndarray,
    tau: np.array,
    wavelength: np.array,
    spectrum_padding: float = 50.0,
    output_path: Optional[str] = None,
):
    """Plot FROG trace as a function of frequency and delay (tau).

    Args:
        spectrogram: FROG signal i.e. spectrogram to plot. Shape should be (num_wavelength, num_time_delay).
        tau: time delay. Shape should be (num_time_delay,)
        wavelength: wavelengths in nm. Shape should be (num_wavelength,)
        spectrum_padding (optional): plot this additional range of values outside the target spectrum range in nm.
            Defaults to 50 - i.e. plot from 750 - 50 = 700 nm to 850 + 50 = 900 nm.
        output_path (optional): path where to save the file. If None, do not save the figure. Defaults to None.
    """
    assert tau.ndim == wavelength.ndim == 1, (
        "Time delays and wavelengths should be 1D arrays."
        f"Got {tau.shape} and {wavelength.shape}."
    )

    assert spectrogram.shape[0] == wavelength.shape[0], (
        "Input spectrogram frequency dimension should match wavelenghts."
        f"Got {spectrogram.shape} and {wavelength.shape}."
    )

    assert spectrogram.shape[1] == tau.shape[0], (
        "Input spectrogram second dimension should match time delays."
        f"Got {spectrogram.shape} and {tau.shape}."
    )

    plt.figure(figsize=PLEASANT_FIG_SIZE)
    plot = plt.pcolormesh(tau, wavelength, spectrogram, shading="auto", cmap="jet")
    cbar = plt.colorbar(plot)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Normalized Intensity")
    plt.xlabel(r"Delay $\tau$ (fs)")
    plt.ylabel("Wavelength (nm)")
    plt.xlim(tau.min() - 0.1, tau.max() + 0.1)
    plt.ylim(
        (SPECTRUM_RANGE[0] - spectrum_padding, SPECTRUM_RANGE[1] + spectrum_padding)
    )
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
