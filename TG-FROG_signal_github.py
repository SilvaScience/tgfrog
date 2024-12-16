#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:27:20 2024

@author: isabella
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt


#%% Étape 1: Création d'un pulse gaussien
def generate_gaussian_pulse(omega, omega_0, phi_coeff, sigma_omega):
    """
    Création d'un pulse Gaussien dans le domaine des fréquences 
    Les paramètres:
        omega : array --> Fréquences angulaires (rad/fs)
        omega_0 : float --> Fréquence centrale (rad/fs)
        sigma_omega : float --> Largeur spectrale (rad/fs)
        phi_coeff : dict --> Dictionnaire des coefficients du polynôme de phase
        {0: phi_0 (rad), 1:phi_1 (fs), 2: phi_2 (fs^2),...}
    La fonction retourne E_omega : array --> Champ électrique dans le domaine des fréquences
    """
    
    # Calcul de l'amplitudes gaussienne centrée sur omega_0
    E_omega = np.exp(-(omega - omega_0)**2 / (2 * sigma_omega**2))
    
    # Calcul de la phase spectrale totale
    phi = np.sum([coeff * (omega - omega_0)**order
                  for order, coeff in phi_coeff.items()], axis=0)
    
    return E_omega* np.exp(1j * phi)

# Graphique de l'intensité spectral en fonction de la phase

def plot_spectral_intensity_and_phase(omega, E_omega):
    """
    Visualise l'intensité spectrale et la phase du pulse sur le même graphique
    en fonction de la longueur d'onde
    """
    # Conversion des fréquences en longueurs d'onde
    c = 3e8 # m/s
    mask = np.abs(omega) > 1e-10 # Évite la division par 0
    wavelength = np.zeros_like(omega)
    wavelength[mask] = 2 * np.pi * c * 1e-15 / omega[mask] * 1e9  # conversion en nm
    
    # Calcul de l'intensité et de la phase
    intensity = np.abs(E_omega)**2
    phase = np.unwrap(np.angle(E_omega)) # rad
    
    # Filtrage entre 750-850 nm et normalisation de l'intensité
    valid_mask = (wavelength >= 750) & (wavelength <= 850) & mask
    wavelength = wavelength[valid_mask]
    intensity = intensity[valid_mask] / np.max(intensity[valid_mask])   # Normalisation 
    phase = phase[valid_mask] - np.mean(phase[valid_mask])  # Phase relative
    
    # Création de la figure avec deux axes y partageant le même axe x
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot de l'intensité sur l'axe gauche
    ax1.plot(wavelength, intensity, 'b-', linewidth=2, label='Intensité')
    ax1.set_xlabel('Longueur d\'onde (nm)')
    ax1.set_ylabel('Intensité normalisée')
    
    # Plot de la phase sur l'axe droit
    ax2.plot(wavelength, phase, 'r--', linewidth=2, label='Phase')
    ax2.set_ylabel('Phase (rad)')
    
    # Configuration des limites et grille
    ax1.set_xlim(750, 850)
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(-5, 5)
    
    # Ajout des légendes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    
    plt.title('Intensité spectrale et phase')
    plt.grid(True)
    plt.show()
    


#%% Étape 2 : Génération des trois champs
def generate_three_fields(omega, E_omega, omega_0):
    """
    Création de trois champs avec différentes phases spectrales
    """
    # Définition des chirps (tous à 0 dans cette version, possible de modifier)
    phi2_1 = 0 #fs^2
    phi2_2 = 0 #fs^2
    phi2_3 = 0 #fs^2
    
    # Normalisation des fréquences pour éviter l'overflow
    delta_omega = omega - omega_0   # rad/fs
    delta_omega_norm = delta_omega * 1e-1  # Réduction de l'échelle
    
    E1_omega = E_omega * np.exp(1j * 0.5 * phi2_1 * delta_omega_norm**2)
    E2_omega = E_omega * np.exp(1j * 0.5 * phi2_2 * delta_omega_norm**2)
    E3_omega = E_omega * np.exp(1j * 0.5 * phi2_3 * delta_omega_norm**2)
    
    # Champ dans l'espace des phases
    E1_t = ifft(ifftshift(E1_omega))
    E2_t = ifft(ifftshift(E2_omega))
    E3_t = ifft(ifftshift(E3_omega))

    
    return E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega

#%% Étape 3: Calculer le signal TG-FROG
def tg_frog(E1_t, E2_t, E3_t, t, tau, omega):
    """
    Calcule le signal TG-FROG
    """
    N = len(t)
    signal_t_tau = np.zeros((N, len(tau)), dtype=complex)
    
    for i, delay in enumerate(tau):
        # Calculer le nombre de points de décalage
        shift_points = int(delay / (t[1] - t[0]))
        
        # Décaler E2_t en utilisant np.roll()
        E2_delayed_t = np.roll(E2_t, shift_points)

        
        # Calcul du signal TG-FROG
        signal_t_tau[:, i] = E1_t * np.conj(E2_delayed_t) * E3_t
    
    return signal_t_tau

#%% Étape 4: Résultats

def plot_frog_trace(signal_t_tau, t, tau, omega, title="Trace FROG"):
    """
    Trace la FROG en fonction du délai tau et de la longueur d'onde.
    """
    # Calcul du spectrogramme
    signal_freq_tau = fftshift(fft(signal_t_tau, axis=0), axes=0)
    intensity = np.abs(signal_freq_tau)**2
    
    # Calcul correct des longueurs d'onde avec gestion de la division par zéro
    c = 3e8  # m/s
    freq = omega * 1e15  # Conversion en Hz
    
    # Créer un masque pour éviter la division par zéro
    nonzero_mask = freq != 0
    wavelength = np.zeros_like(freq)  # Initialiser avec des zéros
    wavelength[nonzero_mask] = 2 * np.pi * c / freq[nonzero_mask] * 1e9  # Conversion en nm
    
    # Filtrer les valeurs non physiques entre 700-900 nm
    valid_mask = (wavelength > 0) & (wavelength >= 700) & (wavelength <= 900)
    wavelength_plot = wavelength[valid_mask]
    intensity_plot = intensity[valid_mask, :]
    
    # vérification des données pour s'assurer que la plage choisie contient des données
    if len(wavelength_plot) == 0:
        print("Erreur: Aucune donnée dans la plage de longueurs d'onde spécifiée")
        print(f"Plage de longueurs d'onde calculée: {np.min(wavelength)} - {np.max(wavelength)} nm")
        return
    
    # Normalisation
    intensity_plot = intensity_plot / np.max(intensity_plot)
    
    # Tri pour assurer la monotonie
    sort_idx = np.argsort(wavelength_plot)
    wavelength_plot = wavelength_plot[sort_idx]
    intensity_plot = intensity_plot[sort_idx, :]
    

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(tau, wavelength_plot, intensity_plot, 
                   shading='auto', cmap='jet')
    plt.colorbar(label="Intensité normalisée")
    plt.xlabel("Délai τ (fs)")
    plt.ylabel("Longueur d'onde (nm)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

#%% Fonction pour les paramètres de la simulation    
    
def main():
    # Paramètres de simulation
    N = 2048
    dt = 1.0 # fs
    t = np.linspace(-N/2 * dt, N/2 * dt, N) # fs
    
    # Calcul de la fréquence centrale pour 800nm
    c = 3e8 # m/s
    lambda_0 = 800e-9 # m 
    omega_0 = 2 * np.pi * c / lambda_0 * 1e-15 # rad/fs
    
    # Calcul de l'axe des fréquences
    dw = 2 * np.pi/ (N * dt)    # rad/fs
    omega = dw * (np.arange(N) - N/2)   #rad/fs
    
    # Largeur spectrale 
    delta_lambda = 40e-9  # m 
    delta_omega_width = 2 * np.pi * c * delta_lambda/ (lambda_0**2) * 1e-15 # rad/fs
    sigma_omega = delta_omega_width / 2.335 # rad/fs 
    
    # Définition des paramètres du pulse initial
    phi = {0: 0, 1: 0, 2: 500}  # Coefficients de phase phi0: rad, phi1: fs, phi2: fs^2
    
    # Génération du pulse
    E_omega = generate_gaussian_pulse(omega, omega_0, phi, sigma_omega)
    
    # Visualisation
    plot_spectral_intensity_and_phase(omega, E_omega)
    
    E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega = generate_three_fields(omega, E_omega, omega_0)
    
    tau = np.linspace(-200, 200, 400) # fs
    signal_t_tau = tg_frog(E1_t, E2_t, E3_t, t, tau, omega)
    plot_frog_trace(signal_t_tau, t, tau, omega)
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    

    
    
    
    



    


    




