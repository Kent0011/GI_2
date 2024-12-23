import zern.zern_core as zern   # import the main library

# import logging
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import itertools

plt.rc('font', family='serif')
plt.rc('text', usetex=False)
cmap = 'jet'

# Parameters
N = 200      # Number of pixels
N_zern = 50
rho_max = 1.0
randgen = RandomState(12345)  # random seed
lambda0 = 0.65

# [0] Construct the coordinates and the aperture mask - simple circ
x = np.linspace(-rho_max, rho_max, N)
xx, yy = np.meshgrid(x, x)
rho = np.sqrt(xx ** 2 + yy ** 2)
theta = np.arctan2(xx, yy)
aperture_mask = rho <= rho_max
rho, theta = rho[aperture_mask], theta[aperture_mask]






# ゼルニケ多項式第4項のみの波面を算出 (球面波)
z = zern.Zernike(mask=aperture_mask)
z.create_model_matrix(rho, theta, n_zernike=N_zern, mode='Jacobi', normalize_noll=False)

Np, Nz = z.model_matrix_flat.shape
print(f"Zernike class holds a model matrix of shape ({Np}, {Nz})")
print(f"Np = {Np} is the number of non-zero entries in our aperture")
print(f"Nz = {Nz} is the total number of Zernike polynomials modelled!!")

coef = np.zeros(N_zern)
coef[4] = lambda0 / 4
phase_map = z.get_zernike(coef)





# 波面の可視化
fig, ax = plt.subplots(1, 1)
MAX = max(np.max(phase_map), -np.min(phase_map))
img = ax.imshow(phase_map, cmap=cmap, vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.show()






# y方向の傾きを計算
phase_map_y = np.zeros((200,200))

for i in range(1,N):
    for j in range(N):
        phase_map_y[i][j] = phase_map[i][j] - phase_map[i-1][j]

l = np.array([20,60,100,140,180])

# for i in itertools.product(l,repeat=2):
    # print("Y -- i:",i[0],"j:",i[1]," = ", phase_map_y[i[0]][i[1]])
    
fig, ax = plt.subplots(1, 1)
MAX = max(np.max(phase_map_y), -np.min(phase_map_y))
img = ax.imshow(phase_map_y, cmap=cmap, vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.show()





# x方向の傾きを計算
phase_map_x = np.zeros((200,200))

for i in range(N):
    for j in range(1,N):
        phase_map_x[i][j] = phase_map[i][j] - phase_map[i][j-1]

l = np.array([20,60,100,140,180])

for i in itertools.product(l,repeat=2):
    print(i[0],",",i[1],",",phase_map_x[i[0]][i[1]],",", phase_map_y[i[0]][i[1]])

fig, ax = plt.subplots(1, 1)
MAX = max(np.max(phase_map_x), -np.min(phase_map_x))
img = ax.imshow(phase_map_x, cmap=cmap, vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.show()
# z.N_total

