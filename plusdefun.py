from fun import *
import numpy as np
from matplotlib import pyplot as plt

x=np.linspace(-10,10,1000)
a=3
b=1

plt.plot(x,np.real(lol(x,a,b)),label='Real')
plt.plot(x,np.imag(lol(x,a,b)),label='Imag')
plt.legend()
plt.show()