import csv
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import zern.zern_core as zern
from numpy.random import RandomState

PX = 7.5
MLA_F = 14200

df = pd.read_csv("/Users/kent/Desktop/GI_2/src/zernike/dot500.csv", header=None)

x = [20, 60, 100, 140, 180]
y = x.copy()
d_zero = itertools.product(x,y)
d_zero = pd.DataFrame(d_zero)

ans =  (df - d_zero) * PX / MLA_F
ans.columns = ["delta_y", "delta_x"]
d_zero.columns = ["y", "x"]

df = pd.concat([d_zero, ans], axis=1)
df = df[['x', 'y', 'delta_x', 'delta_y']]
df = df.sort_values(by=['y','x'], ascending=True)

new_x_coord = np.linspace(0, 200, 200, endpoint=False)
new_y_coord = np.linspace(0, 200, 200, endpoint=False)
xx, yy = np.meshgrid(new_x_coord, new_y_coord)

knew_xy_coord = df[['x', 'y']].values
knew_values = df['delta_x'].values
result_x = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method='cubic')

fig = plt.figure()
ax = fig.add_subplot(111)
MAX = 0.2
img = ax.imshow(result_x, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

knew_xy_coord = df[['x', 'y']].values
knew_values = df['delta_y'].values
result_y = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method='cubic')

fig = plt.figure()
ax = fig.add_subplot(111)
MAX = max(np.max(result_y), -np.min(result_y))
img = ax.imshow(result_y, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

W_x = np.zeros((200,200))
W_x[20][20] = 0.176

for i in range(20,180):
    W_x[i+1][20] = W_x[i][20] + result_y[i+1][20]
    
for j in range(20,180):
    for i in range(20,180):
        W_x[i][j+1] = W_x[i][j] + result_x[i][j+1]

fig = plt.figure()
ax = fig.add_subplot(111)
MAX = 0.2
img = ax.imshow(W_x, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

W_y = np.zeros((200,200))
W_y[20][20] = 0.176

for j in range(20,180):
    W_y[20][j+1] = W_y[20][j] + result_x[20][j+1]
    
for i in range(20,180):
    for j in range(20,180):
        W_y[i+1][j] = W_y[i][j] + result_y[i+1][j]

fig = plt.figure()
ax = fig.add_subplot(111)
MAX = 0.2
img = ax.imshow(W_y, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

W = (W_x + W_y) /2

plt.rc('font', family='serif')
plt.rc('text', usetex=False)
cmap = 'jet'

N = 200
N_zern = 50
rho_max = 1.0
randgen = RandomState(12345)
lambda0 = 0.65

x = np.linspace(-rho_max, rho_max, N)
xx, yy = np.meshgrid(x, x)
rho = np.sqrt(xx ** 2 + yy ** 2)
theta = np.arctan2(xx, yy)
aperture_mask = rho <= rho_max
rho, theta = rho[aperture_mask], theta[aperture_mask]

z = zern.Zernike(mask=aperture_mask)
z.create_model_matrix(rho, theta, n_zernike=N_zern, mode='Jacobi', normalize_noll=False)

Np, Nz = z.model_matrix_flat.shape

coef = np.zeros(N_zern)
coef[4] = lambda0 / 4
phase_map = z.get_zernike(coef)

fig = plt.figure()
ax = fig.add_subplot(111)
MAX = max(np.max(phase_map), -np.min(phase_map))
img = ax.imshow(phase_map, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
img = ax.imshow(W, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
img = ax.imshow(W-phase_map, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()