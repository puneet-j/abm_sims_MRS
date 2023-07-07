import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 1.0, 100)
# y1 = np.power(5.0, [-2.0*a for a in x])
y2 = np.power(20.0, [-7.0*a for a in x])
y3 = np.power(5.0, [-2.0*a for a in x])

plt.figure(1)
# plt.plot(x,1-y1,'r.')
plt.plot(x,1-y2,'b.')
plt.plot(x,1-y3,'k.')
plt.show()