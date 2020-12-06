import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0, 10, 1000)
signal = np.exp((-2 - (2 * np.pi * 1j)) * time)
imaginary_part = signal.imag
plt.plot(time, imaginary_part)
plt.ylabel("Signal (Im(x(t)))")
plt.xlabel("Time (t)")
plt.title("Time vs. Imaginary part of the signal x(t)=e^(-2(1+(j * pi))t)")
plt.show()
