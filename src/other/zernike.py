import csv
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

PX = 7.5

df = pd.read_csv("/Users/kent/Desktop/GI_2/image/testforZERN_dots.csv", header=None)

x = [20, 60, 100, 140, 180]
y = x.copy()

d_zero = itertools.product(x,y)

d_zero = pd.DataFrame(d_zero)

ans =  (df - d_zero) * PX / 8700
ans.columns = ["delta_y", "delta_x"]
d_zero.columns = ["y", "x"]

df = pd.concat([d_zero, ans], axis=1)
df = df[['x', 'y', 'delta_x', 'delta_y']]

df = df.sort_values(by=['y','x'], ascending=True)

print(df)

new_x_coord = np.linspace(0, 200, 200, endpoint=False)
new_y_coord = np.linspace(0, 200, 200, endpoint=False)

xx, yy = np.meshgrid(new_x_coord, new_y_coord)


knew_xy_coord = df[['x', 'y']].values
knew_values = df['delta_x'].values

result_x = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method='cubic')

knew_xy_coord = df[['x', 'y']].values
knew_values = df['delta_y'].values

result_y = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method='cubic')

W_x = np.zeros((200,200))

for i in range(20,180):
    W_x[i+1][20] = W_x[i][20] + result_y[i+1][20]
    
for j in range(20,180):
    for i in range(20,180):
        W_x[i][j+1] = W_x[i][j] + result_x[i][j+1]
        
fig = plt.figure()
ax = fig.add_subplot(111)
MAX = max(np.max(W_x), -np.min(W_x))
img = ax.imshow(W_x, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()



W_y = np.zeros((200,200))


for j in range(20,180):
    W_y[20][j+1] = W_y[20][j] + result_x[20][j+1]
    
for i in range(20,180):
    for j in range(20,180):
        W_y[i+1][j] = W_y[i][j] + result_y[i+1][j]
        
fig = plt.figure()
ax = fig.add_subplot(111)
MAX = max(np.max(W_y), -np.min(W_y))
img = ax.imshow(W_y, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()

W = (W_x + W_y) /2


fig = plt.figure()
ax = fig.add_subplot(111)
MAX = max(np.max(W), -np.min(W))
img = ax.imshow(W, cmap="jet", vmin=-MAX, vmax=MAX)
plt.colorbar(img)
plt.gca().set_ylim(199, 0)
plt.show()
