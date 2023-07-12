import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 1.0, 100)
# y1 = np.power(5.0, [-2.0*a for a in x])
y2 = np.power(10.0, [-2.0*a for a in x])
# y3 = np.power(5.0, [-2.0*a for a in x])

plt.figure(1)
# plt.plot(x,1-y1,'r.')
plt.plot(x,1-y2,'b.')
# plt.plot(x,1-y3,'k.')
plt.show()


# # # Import matplotlib, numpy and math
# import matplotlib.pyplot as plt
# import numpy as np
# # import math
  
# x = np.linspace(0, 1, 200)
# z = 1/(1 + np.exp(-10*x))
  
# plt.plot(x, z)
# plt.xlabel("x")
# plt.ylabel("Sigmoid(X)")
  
# plt.show()