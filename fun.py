import numpy as np
import scipy as sp


def lol(x,a,b):
    '''
    C'est drole les fonctions
    '''
    y=b*np.exp(1j*x*a)+b*np.exp(-1j*x*a)
    return y