from sys import intern
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from tgfrog.plotting_tools import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
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
        "Input spectrogram frequency dimension should match wavelengths."
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


def plot_estimated_and_target_spectrogram(
    estimated_spectrogram: np.ndarray,
    target_spectrogram: np.ndarray,
    tau: np.array,
    wavelength: np.array,
    spectrum_padding: float = 50.0,
    output_path: Optional[str] = None,
):
    """Plot an estimated FROG spectrogram, a target spectrogram and the difference between them.

    Args:
        estimated_spectrogram: estimated spectrogram. Shape should be (num_wavelength, num_time_delay).
        target_spectrogram: target spectrogram. Shape should be (num_wavelength, num_time_delay).
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

    assert estimated_spectrogram.shape[0] == wavelength.shape[0], (
        "Input spectrogram frequency dimension should match wavelengths."
        f"Got {estimated_spectrogram.shape} and {wavelength.shape}."
    )

    assert estimated_spectrogram.shape[1] == tau.shape[0], (
        "Input spectrogram second dimension should match time delays."
        f"Got {estimated_spectrogram.shape} and {tau.shape}."
    )

    assert estimated_spectrogram.shape == target_spectrogram.shape, (
        "Estimated spectrogram shape does not match target."
    )

    fig = plt.figure(figsize=(PLEASANT_FIG_SIZE[0] * 3, PLEASANT_FIG_SIZE[1]))

    ax1 = fig.add_subplot(131)

    estimated_plot = ax1.pcolormesh(tau, wavelength, estimated_spectrogram, cmap="jet")
    ax1.set_xlabel(r"Delay $\tau$ (fs)")
    ax1.set_ylabel("Wavelength (nm)")
    ax1.set_xlim(tau.min() - 0.1, tau.max() + 0.1)
    ax1.set_ylim(
        (SPECTRUM_RANGE[0] - spectrum_padding, SPECTRUM_RANGE[1] + spectrum_padding)
    )
    ax1.set_title("Estimated Spectrogram")

    ax2 = fig.add_subplot(132)
    ax2.pcolormesh(tau, wavelength, target_spectrogram, cmap="jet")
    ax2.set_xlabel(r"Delay $\tau$ (fs)")
    ax2.set_ylabel("Wavelength (nm)")
    ax2.set_xlim(tau.min() - 0.1, tau.max() + 0.1)
    ax2.set_ylim(
        (SPECTRUM_RANGE[0] - spectrum_padding, SPECTRUM_RANGE[1] + spectrum_padding)
    )
    ax2.set_title("Target Spectrogram")

    ax3 = fig.add_subplot(133)
    difference_plot = ax3.pcolormesh(tau, wavelength, estimated_spectrogram - target_spectrogram, cmap="jet")
    ax3.set_xlabel(r"Delay $\tau$ (fs)")
    ax3.set_ylabel("Wavelength (nm)")
    ax3.set_xlim(tau.min() - 0.1, tau.max() + 0.1)
    ax3.set_ylim(
        (SPECTRUM_RANGE[0] - spectrum_padding, SPECTRUM_RANGE[1] + spectrum_padding)
    )
    ax3.set_title("Difference")

    cax = ax1.inset_axes([-0.18, 0.1, 0.02, 0.8])
    cbar1 = fig.colorbar(estimated_plot, cax=cax, location='left', shrink=0.8)
    cbar1.ax.set_ylabel("Normalized Intensity")

    # Colorbar for the third axis
    cax2 = ax3.inset_axes([1.06, 0.1, 0.02, 0.8])
    cbar2 = fig.colorbar(difference_plot, cax=cax2, location='right', shrink=0.8)
    cbar2.ax.set_ylabel('Difference in normalized intensity 3')

    # plt.tight_layout()
    fig.subplots_adjust(left=0.08)  # Make space on the right for the first colorbar
    fig.subplots_adjust(right=0.93)  # Make space on the right for the first colorbar
    fig.subplots_adjust(wspace=0.10)

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
