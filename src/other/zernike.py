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

ans =  (df - d_zero)
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
knew_values = df['delta_y'].values

result = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method='cubic')

# グラフ表示
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
ax.contourf(xx, yy, result, cmap='jet')
plt.gca().set_ylim(199, 0)
plt.show()