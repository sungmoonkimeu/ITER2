import numpy as np

shape = (2,2,8)
identity_3d = np.zeros(shape)
np.einsum('iij->ij', identity_3d)[:] = 1


