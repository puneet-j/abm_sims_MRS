import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from params import PWR_RECRUIT_TO_TS,ADD_RECRUIT_QUAL,MULTIPLIER_RECRUIT_QUAL

# PWR_RECRUIT_TO_TS = 1.0
# ADD_RECRUIT_QUAL = 0.7
# MULTIPLIER_RECRUIT_QUAL = -10.0

x = np.linspace(0.01, 1.0, 200)
# # y1 = np.power(5.0, [-2.0*a for a in x])
# y2 = np.power(10.0, [-2.0*a for a in x])
# # y3 = np.power(5.0, [-2.0*a for a in x])
stay_attach_prob_due_to_qual = x**PWR_RECRUIT_TO_TS
RECRUIT_qual_factor = ADD_RECRUIT_QUAL/(ADD_RECRUIT_QUAL + np.exp(MULTIPLIER_RECRUIT_QUAL*x))

p = 0
recruit_to_ts0 = stay_attach_prob_due_to_qual*(1.0 - p*RECRUIT_qual_factor)
recruit_to_o0 = (1.0 - stay_attach_prob_due_to_qual)*(1.0 - p*stay_attach_prob_due_to_qual)
recruit_to_r0 = 1.0 - recruit_to_ts0 - recruit_to_o0

plt.figure(1)
# # plt.plot(x,1-y1,'r.')

plt.plot(x, recruit_to_ts0,'b.')
plt.plot(x, recruit_to_o0, 'r.')
plt.plot(x, recruit_to_r0, 'k.')

plt.legend(['recruit_to_ts0', 'recruit_to_o0', 'recruit_to_r0'])
plt.title("When Recruit to Observe is 0:  2% times (BINOMIAL_COEFF_RECRUIT_TO_OBSERVE = 0.98)")


p = 1
recruit_to_ts1 = stay_attach_prob_due_to_qual*(1.0 - p*RECRUIT_qual_factor)
recruit_to_o1 = (1.0 - stay_attach_prob_due_to_qual)*(1.0 - p*stay_attach_prob_due_to_qual)
recruit_to_r1 = 1.0 - recruit_to_ts1 - recruit_to_o1

plt.figure(2)
# # plt.plot(x,1-y1,'r.')

plt.plot(x, recruit_to_ts1,'b.')
plt.plot(x, recruit_to_o1, 'r.')
plt.plot(x, recruit_to_r1, 'k.')

plt.legend(['recruit_to_ts1', 'recruit_to_o1', 'recruit_to_r1'])

plt.title("When Recruit to Observe is 1:  98% times (BINOMIAL_COEFF_RECRUIT_TO_OBSERVE = 0.98)")
  
plt.show()

