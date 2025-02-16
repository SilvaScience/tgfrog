from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tgfrog.plotting_tools import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from tgfrog.utils.constants import (METER_TO_NANOMETER, OMEGA_TO_HERTZ,
                                    SPECTRUM_RANGE, SPEED_OF_LIGHT, TWOPI)

plt.style.use(PLOT_STYLE_PATH)
WAVELENGTH_CUTOFF = 1e-10


def plot_signal_spectrum(
    omega: np.array,
    e1_omega: np.array,
    e2_omega: np.array,
    e3_omega: np.array,
    chirps: Tuple[float, float, float] = (0, 0, 0),
    output_path: Optional[str] = None,
) -> None:
    """Plot signal spectrum as a function of three incoming electric fields.

    Args:
        omega: frequency. Should be of size (num_freq,)
        e1_omega: first electric field as a function of omega with complex values. Should be of size (num_freq,)
        e2_omega: second electric field as a function of omega with complex values. Should be of size (num_freq,)
        e3_omega: third electric field as a function of omega with complex values. Should be of size (num_freq,)
        chirps: parameters describing the chirps in fs^2. Defaults to (0, 0, 0)
        output_path (optional): path where to save the file. If None, do not save the figure. Defaults to None.
    """
    assert omega.shape == e1_omega.shape, (
        "Electric field dimension should match input frequencies."
        f"Got {omega.shape} for frequencies and {e1_omega.shape} for first electric field."
    )

    assert omega.shape == e2_omega.shape, (
        "Electric field dimension should match input frequencies."
        f"Got {omega.shape} for frequencies and {e2_omega.shape} for second electric field."
    )

    assert omega.shape == e3_omega.shape, (
        "Electric field dimension should match input frequencies."
        f"Got {omega.shape} for frequencies and {e3_omega.shape} for first electric field."
    )

    # combine signals - note that the second field in complex conjugated (no effect if the field is real)
    e_signal = e1_omega * np.conj(e2_omega) * e3_omega

    # compute intensity and phase of combined signals
    intensity = np.abs(e_signal) ** 2
    phase = np.unwrap(np.angle(e_signal))

    # convert omega frequencies in wavelength
    mask = np.abs(omega) > WAVELENGTH_CUTOFF
    wavelength = np.zeros_like(omega)
    wavelength[mask] = (
        TWOPI * SPEED_OF_LIGHT * METER_TO_NANOMETER / (omega[mask] * OMEGA_TO_HERTZ)
    )

    valid_mask = (
        (wavelength >= SPECTRUM_RANGE[0]) & (wavelength <= SPECTRUM_RANGE[1]) & mask
    )
    wavelength = wavelength[valid_mask]
    intensity = intensity[valid_mask] / np.max(intensity[valid_mask])
    phase = phase[valid_mask] - np.mean(phase[valid_mask])

    # create the plot
    fig, ax1 = plt.subplots(figsize=PLEASANT_FIG_SIZE)
    ax2 = ax1.twinx()

    ax1.plot(wavelength, intensity, "b-", linewidth=2, label="Intensity")
    ax1.set_xlabel("Wavelength (nm)", fontsize=14)
    ax1.set_ylabel("Normalized Intensity", fontsize=14)

    ax2.plot(wavelength, phase, "r--", linewidth=2, label="Phase")
    ax2.set_ylabel("Phase (rad)", fontsize=14)

    ax1.set_xlim(*SPECTRUM_RANGE)
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(-5, 5)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12)

    chirp_text = (
        r"$\phi_{12}$="
        f"{chirps[0]},"
        r" $\phi_{22}$="
        f"{chirps[1]},"
        r" $\phi_{23}$="
        f"{chirps[2]}"
        r" fs$^2$"
    )
    plt.title(f"Combined signal E1 x E2* × E3 - {chirp_text}", fontsize=14)
    # plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
