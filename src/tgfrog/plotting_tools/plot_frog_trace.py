from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift

from tgfrog.plotting_tools import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from tgfrog.utils.constants import (METER_TO_NANOMETER, OMEGA_TO_HERTZ,
                                    SPECTRUM_RANGE, SPEED_OF_LIGHT, TWOPI)

plt.style.use(PLOT_STYLE_PATH)


def plot_frog_trace(
    signal_t_tau: np.ndarray,
    tau: np.array,
    omega: np.array,
    spectrum_window: float = 50.0,
    output_path: Optional[str] = None,
):
    """Plot FROG trace as a function of frequency and delay (tau).

    Args:
        signal_t_tau: FROG signal. Shape should be (num_freq, num_time_delay).
        tau: time delay. Shape should be (num_time_delay,)
        omega: frequencies. Shape should be (num_freq,)
        spectrum_window (optional): plot this additional range of values outside the target spectrum range in nm.
            Defaults to 50 - i.e. plot from 750 - 50 = 700 nm to 850 + 50 = 900 nm.
        output_path (optional): path where to save the file. If None, do not save the figure. Defaults to None.
    """
    assert tau.ndim == omega.ndim == 1, (
        f"Time delays and frequencies should be 1D arrays."
        "Got {tau.shape} and {omega.shape}."
    )

    assert signal_t_tau.shape[0] == omega.shape[0], (
        "Input signal first dimension should match frequencies."
        f"Got {signal_t_tau.shape} and {omega.shape}."
    )

    assert signal_t_tau.shape[1] == omega.shape[1], (
        "Input signal second dimension should match time delays."
        f"Got {signal_t_tau.shape} and {tau.shape}."
    )

    # compute spectrogram
    signal_freq_tau = fftshift(fft(signal_t_tau, axis=0), axes=0)
    intensity = np.abs(signal_freq_tau) ** 2

    # compute wavelengths
    freq = omega * OMEGA_TO_HERTZ  # convert to Hz
    nonzero_mask = freq != 0
    wavelength = np.zeros_like(freq)
    wavelength[nonzero_mask] = (
        TWOPI * SPEED_OF_LIGHT * METER_TO_NANOMETER / freq[nonzero_mask]
    )

    # keep only wavelengths in the target spectrum range within
    valid_mask = (wavelength >= SPECTRUM_RANGE[0] - spectrum_window) & (
        wavelength <= SPECTRUM_RANGE[1] + spectrum_window
    )
    wavelength_plot = wavelength[valid_mask]
    intensity_plot = intensity[valid_mask, :]

    assert len(wavelength_plot) > 0, "No values are in the specified range."

    # Normalization
    intensity_plot = intensity_plot / np.max(intensity_plot)
    sort_idx = np.argsort(wavelength_plot)
    wavelength_plot = wavelength_plot[sort_idx]
    intensity_plot = intensity_plot[sort_idx, :]

    plt.figure(figsize=PLEASANT_FIG_SIZE)
    plot = plt.pcolormesh(
        tau, wavelength_plot, intensity_plot, shading="auto", cmap="jet"
    )
    cbar = plt.colorbar(plot)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Normalized Intensity", size=14, labelpad=12)
    plt.xlabel(r"Delay $\tau$ (fs)", fontsize=14)
    plt.ylabel("Wavelength (nm)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
