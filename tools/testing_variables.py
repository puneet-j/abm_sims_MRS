import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 1.0, 200)
# # y1 = np.power(5.0, [-2.0*a for a in x])
# y2 = np.power(10.0, [-2.0*a for a in x])
# # y3 = np.power(5.0, [-2.0*a for a in x])
y = x**5.0

val = 0.5
mul = -5.0
# x = np.linspace(0.01, 1.0, 200)
z = val/(val + + np.exp(mul*x))
# z = 1.0/(1.0 + np.exp(mul*x + val))

plt.figure(1)
# # plt.plot(x,1-y1,'r.')
plt.plot(x, y,'b.')
plt.plot(x, z, 'r.')

# # plt.plot(x,1-y3,'k.')


# # # Import matplotlib, numpy and math
# import matplotlib.pyplot as plt
# import numpy as np
# # import math


# plt.figure(2)

# plt.xlabel("x")
plt.ylabel("Power (Blue) = TS vs Rest and Sigmoid (Red) = (Dance Time)")

plt.title("Power Blue and Sigmoid Red")
  
plt.show()

# print('\n 0: ')
# print(1.0 - np.power(10.0, -2.0*0.0))
# print(1/(1 + np.exp(-10*0.0)))

# print('\n 1: ')
# print(1.0 - np.power(10.0, -2.0*1.0))
# print(1/(1 + np.exp(-10*1.0)))

# print('\n 0.5: ')
# print(1.0 - np.power(10.0, -2.0*0.5))
# print(1/(1 + np.exp(-10*0.5)))