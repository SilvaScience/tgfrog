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
def generate_gaussian_pulse(omega):
    """
    Création d'un pulse Gaussien dans le domaine des fréquences 
    Les paramètres peuvent être modifiés selon les besoins
    """

    sigma_omega = 2e13  #Largeur spectral
    E_omega = np.exp(-omega**2 / (2 * sigma_omega**2))
                 
    return E_omega

#%% Étape 2 : Génération des trois champs
def generate_three_fields(omega, E_omega):
    """
    Création de trois champs avec différentes phases spectrales
    """
    
    phi1 = 0
    phi2 = 1e-24
    phi3 = 5e-25
    
    E1_omega = E_omega * np.exp(1j * phi1 * omega**2)
    E2_omega = E_omega * np.exp(1j * phi2 * omega**2)
    E3_omega = E_omega * np.exp(1j * phi3 * omega**2)
    
    # Champ dans l'espace des phases
    E1_t = ifft(ifftshift(E1_omega))
    E2_t = ifft(ifftshift(E2_omega))
    E3_t = ifft(ifftshift(E3_omega))
    
    
    
    return E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega

#%% Étape 3: Calculer le signal TG-FROG
def tg_frog(E1_t, E2_t, E3_t, t, tau, omega):
    N = len(t)
    signal_t_tau = np.zeros((N, len(tau)), dtype=complex)
    
    for i, delay in enumerate(tau):
        # Création d'un délais tau sur le champ E2
        E2_freq = fft(E2_t)
        E2_delayed = ifft(E2_freq * np.exp(-1j * omega * delay))
        
        # Calcul du signal
        signal_t_tau[:, i] = E1_t * np.conj(E2_delayed) * E3_t
    
    return signal_t_tau

#%% Étape 4: Résultats

def plot_results(t, tau, omega, signal_t_tau, E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega):
    # Conversion des unités
    t_fs = t * 1e15     # Conversion en femtosecondes
    tau_fs = tau * 1e15
    freq_THz = omega / (2 * np.pi) * 1e-12  # Conversion en THz
    
    # figure 1:  t vs tau
    plt.figure(figsize=(10,6), dpi=200)
    signal_t_tau_intensité = np.abs(signal_t_tau)**2
    plt.imshow(signal_t_tau_intensité, aspect="auto", extent=[tau_fs[0], tau_fs[-1], t_fs[0], t_fs[-1]], origin="lower", cmap="turbo")
    plt.colorbar(label="Intensité")
    plt.xlabel('Délais tau (fs)')
    plt.ylabel("Temps t (fs)")
    
    
    # Plot intensités spectrales
    #plt.figure(dpi=200)
    #freq_Thz = freq * 1e-12
    #plt.plot(freq_Thz, np.abs(E1_omega)**2, 'b-', label='E1')
    #plt.plot(freq_Thz, np.abs(E2_omega)**2, 'r-', label='E2')
    #plt.plot(freq_Thz, np.abs(E3_omega)**2, 'g-', label='E3')
    #plt.xlabel('Fréquence (THz)')
    #plt.ylabel('|E(w)|^2')
    #plt.title('Intensité spectrale')
    #plt.legend()
    #plt.grid(True)
    
    
    # Conversion de la fréquence en longueur d'onde
    #c = 3e8
    # Ajuster la fréquence centrale selon la longueur d'onde voulue, ici j'ai choisi l'ultra-violet, donc une longueur d'onde de 300nm
    #freq_central = 1e15     # 1000 THz
    #freq_hz = freq + freq_central
    #valid_indices = freq_hz != 0    # Filtre les valeurs à 0
    #wavelenght = np.zeros_like(freq_hz)
    #wavelenght[valid_indices] = c / freq_hz[valid_indices] * 1e9
    
    
    # Plot signal TG-FROG
    #tau_fs = tau * 1e15
    #plt.figure(dpi=200)
    #frog_signal_norm = frog_signal / np.max(frog_signal)
    #plt.imshow(frog_signal_norm, aspect='auto', extent=[tau_fs[0], tau_fs[-1], wavelenght[-1], wavelenght[0]], origin='lower', cmap='turbo')
    #plt.xlabel('Délais tau (fs)')
    #plt.ylabel('Longueur d\'onde (nm)')
    #plt.xlim(-500, 500)
    #plt.ylim(380, 420)
    #plt.colorbar(label='Intensité normalisée')
    
    #plt.tight_layout()
    #plt.show()



# Exemple d'exécution
"""
Les données ici peuvent être changée selon les besoins
"""
#def main():
    
 #   N = 2000   # Augmenter le nombre de points pour une meilleure résolution
  #  T = 2000e-15    # Fenêtre temporelle modifier au besoin selon le délais
   # dt = T/N
    #t = np.linspace(-T, T, N)
    
    # Fréquence
   # freq = fftshift(np.fft.fftfreq(N, dt))
    
    # Étape 1
    #E_t = generate_gaussian_pulse(t)
    
    # Étape 2
    #E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega = generate_three_fields(t, E_t)
    
    # Étape 3
    #tau = np.linspace(-T/4, T/4, N)
    #frog_signal = tg_frog(E1_t, E2_t, E3_t, t, tau)
    
    # plot
    #plot_results(t, freq, E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega, frog_signal, tau)
    
def main():
    # Paramètres de simulation
    N = 2000
    T = 2000e-15  # Fenêtre temporelle totale
    dt = T/N
    t = np.linspace(-T/2, T/2, N)
    
    # Fréquences angulaires
    omega = 2 * np.pi * fftshift(np.fft.fftfreq(N, dt))
    
    # Génération du pulse dans le domaine spectral
    E_omega = generate_gaussian_pulse(omega)
    
    # Génération des trois champs
    E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega = generate_three_fields(omega, E_omega)
    
    # Calcul du signal pour différents délais
    tau = np.linspace(-T/4, T/4, N)
    signal_t_tau = tg_frog(E1_t, E2_t, E3_t, t, tau, omega)
    
    # Affichage des résultats
    plot_results(t, tau, omega, signal_t_tau, E1_t, E2_t, E3_t, E1_omega, E2_omega, E3_omega)


# Lancer le programme
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    

    
    
    
    



    


    




