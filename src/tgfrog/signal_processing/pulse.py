from typing import List, Tuple

import numpy as np
from scipy.fftpack import ifft, ifftshift

from tgfrog.utils.constants import (NANOMETER_TO_METER, SECOND_TO_FEMTOSECOND,
                                    SPEED_OF_LIGHT, TWOPI)


class FROG_simulator:
    def __init__(
        self, n: int, dt: float = 1.0, lambda_0: float = 800, delta_lambda: float = 40
    ):
        self.n = n
        self.dt = dt
        # time values for delay (tau) as a linspace
        self.t = np.linspace(-n * dt / 2, n * dt / 2, n)
        lambda_0 *= NANOMETER_TO_METER  # convert nm to m
        # get main frequency in fs^-1
        # speed_of_light and lambda_0 are expressed in meter, speed of light in second
        self.omega_0 = (
            TWOPI * SPEED_OF_LIGHT / (lambda_0 * SECOND_TO_FEMTOSECOND)
        )  # in (fs)^-1

        # frequency values
        dw = TWOPI / (self.n * dt)  # in s^-1
        self.omega = dw * (np.arange(self.n) - self.n / 2)

        delta_lambda *= NANOMETER_TO_METER
        # in rad / fs
        delta_omega_width = (
            TWOPI
            * SPEED_OF_LIGHT
            * delta_lambda
            / ((lambda_0**2) * SECOND_TO_FEMTOSECOND)
        )
        self.sigma_omega = (
            delta_omega_width / 2.335
        )  # not sure where this value comes from

    def generate_gaussian_pulse(self, phi_taylor_coefficient: List[float]) -> np.array:
        # gaussian centered on omega_0
        # amplitude is a gaussian
        e_omega = np.exp(
            -((self.omega - self.omega_0) ** 2) / (2 * self.sigma_omega**2)
        )

        # phase is given by the taylor sum from the coefficient input
        phi = 0
        for i, phi_coefficient in enumerate(phi_taylor_coefficient):
            phi += phi_coefficient * (self.omega - self.omega_0) ** i

        return e_omega * np.exp(1j * phi)

    def generate_three_fields(
        self,
        pulse: np.array,
        phi2_1: float,
        phi2_2: float,
        phi2_3: float,
    ) -> Tuple[np.array, np.array, np.array]:
        delta_omega = self.omega - self.omega_0

        # define the fields in phase space Champs dans l'espace des phases
        e1_omega = pulse * np.exp(1j * 0.5 * phi2_1 * delta_omega**2)
        e2_omega = pulse * np.exp(1j * 0.5 * phi2_2 * delta_omega**2)
        e3_omega = pulse * np.exp(1j * 0.5 * phi2_3 * delta_omega**2)

        # map back to temporal space
        e1_t = ifft(ifftshift(e1_omega))
        e2_t = ifft(ifftshift(e2_omega))
        e3_t = ifft(ifftshift(e3_omega))

        return e1_t, e2_t, e3_t

    def compute_tg_frog(
        self,
        phi_taylor_coefficient: List[float],
        chirp_1: float,
        chirp_2: float,
        chirp_3: float,
        tau: np.array,
    ) -> np.ndarray:
        pulse = self.generate_gaussian_pulse(phi_taylor_coefficient)
        e_fields = self.generate_three_fields(pulse, chirp_1, chirp_2, chirp_3)

        signal_t_tau = np.zeros((self.n, len(tau)), dtype=complex)

        for i, delay in enumerate(tau):
            # delay the second electric field by the delay coming from the tau value
            shift_points = int(delay / self.dt)
            delayed_e2 = np.roll(e_fields[1], shift_points)

            # the TG-FROG signal is the product of the three fields
            signal_t_tau[:, i] = e_fields[0] * np.conj(delayed_e2) * e_fields[2]

        return signal_t_tau


def main():
    n_time_step = 2048
    dt = 1.0  # time resolution in fs
    lambda_0 = 800  # central wavelength in nm
    delta_lambda = 40  # spectral width in nm
    frog_simulator = FROG_simulator(
        n=n_time_step, dt=dt, lambda_0=lambda_0, delta_lambda=delta_lambda
    )
    chirp_configuration = [
        (0, 0, 0),  # no chirps
        (200, 200, 200),  # positive and equal Chirps
        (-200, -200, -200),  # negative and equal Chirps
        (100, -100, 0),  # different Chirps
    ]
    phi_coefficients = [0, 0, 500]
    tau = np.linspace(-200, 200, 400)

    signal_t_tau = []

    for chirp in chirp_configuration:
        signal_t_tau.append(
            frog_simulator.compute_tg_frog(phi_coefficients, *chirp, tau=tau)
        )


if __name__ == "__main__":
    main()
