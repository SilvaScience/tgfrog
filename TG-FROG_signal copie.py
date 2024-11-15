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
def generate_gaussian_pulse(t):
    """
    Création d'un pulse Gaussien dans le domaine des fréquences avec une phase spectrale
    Les paramètres peuvent être ajustés selon les besoins
    """

    sigma = 50e-15 # largeur de 50 fs
    E_t = np.exp(-t**2 / (2 * sigma**2))
                 
    return E_t

#%% Étape 2 : Génération des trois champs
def generate_three_fields(t, E_t):
    """
    Création de trois champs avec différentes phases spectrales
    """
    
    chirp1 = 0
    chirp2 = 1e24
    chirp3 = 5e25
    
    E1_t = E_t * np.exp(1j * chirp1 * t**2)
    E2_t = E_t * np.exp(1j * chirp2 * t**2)
    E3_t = E_t * np.exp(1j * chirp3 * t**2)
    
    # Champ dans l'espace des phases
    E1_omega = fftshift(fft(E1_t))
    E2_omega = fftshift(fft(E2_t))
    E3_omega = fftshift(fft(E3_t))
    
    
    
    return E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega

#%% Étape 3: Calculer le signal TG-FROG
def tg_frog(E1_t, E2_t, E3_t, t, tau):
    N = len(t)
    dt = t[1] - t[0]
    frog_signal = np.zeros((N, len(tau)))
    
    # Calculer le vecteur des fréquences angulaires (en rad/s)
    omega = 2 * np.pi * fftshift(np.fft.fftfreq(N, d=dt))
    
    print(f"Plage des délais (fs): {tau[0]*1e15} à {tau[-1]*1e15}")
    
    for i, delay in enumerate(tau):
        # Création d'un délais tau sur le champ E2
        E2_freq = fft(E2_t)
        E2_delayed = ifft(E2_freq * np.exp(-1j * omega * delay))
        
        # Calcul du signal
        signal = E1_t * np.conj(E2_delayed) * E3_t
        spectrum = np.abs(fftshift(fft(signal)))**2
        frog_signal[:, i] = spectrum
    
    return frog_signal

#%% Étape 4: Résultats

def plot_results(t, freq, E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega, frog_signal, tau):
    # plot intensité
    plt.figure(dpi=200)
    t_fs = t * 1e15
    plt.plot(t_fs, np.abs(E1_t)**2, 'b-', label='E1')
    plt.plot(t_fs, np.abs(E2_t)**2, 'r-', label='E2')
    plt.plot(t_fs, np.abs(E3_t)**2, 'g-', label='E3')
    plt.xlabel('Temps (fs)')
    plt.ylabel('|E(t)|^2')
    plt.title('Intensité temporelle')
    plt.legend()
    plt.grid(True)
    
    
    # Plot intensités spectrales
    plt.figure(dpi=200)
    freq_Thz = freq * 1e-12
    plt.plot(freq_Thz, np.abs(E1_omega)**2, 'b-', label='E1')
    plt.plot(freq_Thz, np.abs(E2_omega)**2, 'r-', label='E2')
    plt.plot(freq_Thz, np.abs(E3_omega)**2, 'g-', label='E3')
    plt.xlabel('Fréquence (THz)')
    plt.ylabel('|E(w)|^2')
    plt.title('Intensité spectrale')
    plt.legend()
    plt.grid(True)
    
    
    # Conversion de la fréquence en longueur d'onde
    c = 3e8
    # Ajuster la fréquence centrale selon la longueur d'onde voulue, ici j'ai choisi l'ultra-violet, donc une longueur d'onde de 300nm
    freq_central = 1e15     # 1000 THz
    freq_hz = freq + freq_central
    valid_indices = freq_hz != 0    # Filtre les valeurs à 0
    wavelenght = np.zeros_like(freq_hz)
    wavelenght[valid_indices] = c / freq_hz[valid_indices] * 1e9
    
    
    # Plot signal TG-FROG
    tau_fs = tau * 1e15
    plt.figure(dpi=200)
    frog_signal_norm = frog_signal / np.max(frog_signal)
    plt.imshow(frog_signal_norm, aspect='auto', extent=[tau_fs[0], tau_fs[-1], wavelenght[-1], wavelenght[0]], origin='lower', cmap='turbo')
    plt.xlabel('Délais tau (fs)')
    plt.ylabel('Longueur d\'onde (nm)')
    plt.xlim(-500, 500)
    plt.ylim(380, 420)
    plt.colorbar(label='Intensité normalisée')
    
    plt.tight_layout()
    plt.show()



# Exemple d'exécution
"""
Les données ici peuvent être changée selon les besoins
"""
def main():
    
    N = 2000   # Augmenter le nombre de points pour une meilleure résolution
    T = 2000e-15    # Fenêtre temporelle modifier au besoin selon le délais
    dt = T/N
    t = np.linspace(-T, T, N)
    
    # Fréquence
    freq = fftshift(np.fft.fftfreq(N, dt))
    
    # Étape 1
    E_t = generate_gaussian_pulse(t)
    
    # Étape 2
    E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega = generate_three_fields(t, E_t)
    
    # Étape 3
    tau = np.linspace(-T/4, T/4, N)
    frog_signal = tg_frog(E1_t, E2_t, E3_t, t, tau)
    
    # plot
    plot_results(t, freq, E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega, frog_signal, tau)
    
# Lancer le programme
if __name__ == "__main__":
    main()
    

print("Test github sur mon iMac")
    
    
    
    
    
    

    
    
    
    



    


    




