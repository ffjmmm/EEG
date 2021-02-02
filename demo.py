from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1.0, 2001)
xlow = np.sin(2 * np.pi * 5 * t)
xhigh = np.sin(2 * np.pi * 50 * t)
x = xlow + xhigh

b, a = signal.butter(10, 0.03, 'high')
filtered = signal.filtfilt(b, a, x)


plt.plot(t, filtered)
plt.show()