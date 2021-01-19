import numpy
from numpy import pi
SR = [0.003]

Len_SF = 28
delta_L = 0.00003  # delta L [m]
dq = 2*pi/SR[0]
V_L = arange(delta_L, Len_SF + delta_L, delta_L)
V_q = dq*V_L
n_L = int(Len_SF/delta_L)
L_rem = Len_SF - n_L* delta_L

print("n_L = ",n_L, "L_rem=", L_rem)
print("Len_Sf= ",Len_SF,"=",L_rem+n_L*delta_L )

num_processor = 4
num_list = arange(0,num_processor,1)

spl_V_q = numpy.array_split(V_q,num_processor)

for nn in range(num_processor):
    n_L_list[nn] = int(n_L / num_processor)
    q0_list[nn] = q + dq*n_L_list[0]*nn
    if nn == num_processor - 1:
        n_L_list[nn] = n_L - int(n_L / num_processor) * num_processor-1