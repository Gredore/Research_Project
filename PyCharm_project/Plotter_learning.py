import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


x = np.array([0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
y =  np.array([0.19, 0.12, 0.23, 0.39, 0.32, 0.34, 0.42])
error =  np.array([0.03, 0.06, 0.08, 0.03, 0.02, 0.06, 0.04])

a, b, c = np.polyfit(x, y, 2)

plt.errorbar(x, y, yerr=error, fmt='.k', markersize=8, capsize=10)
plt.plot(x, a*x**2 + b*x + c, color=((0, 0.7, 0.7)), linewidth = 2)

plt.xlabel('Fraction of whole dataset', fontsize=20)
plt.ylabel('MCC', fontsize=20)

plt.savefig('../ModelALearningCurve.png', format='png', dpi=200)
plt.show()
