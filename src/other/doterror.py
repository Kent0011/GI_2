import csv
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

PX = 7.5

df = pd.read_csv("/Users/kent/Desktop/GI_2/image/testforZERN_dots.csv", header=None)

x = [20, 60, 100, 140, 180]
y = x.copy()

d_zero = itertools.product(x,y)

d_zero = pd.DataFrame(d_zero)

ans =  (df - d_zero)*PX
ans.columns = ["delta_x", "delta_y"]

dW = {}

for i in d_zero.itertuples(index=True):
    dW[(i._1 - 100, 100 - i._2)] = (ans.iloc[i.Index,0], ans.iloc[i.Index,1])
    
for zero, ans in dW.items():
    print(f'dW/dx{zero} = {ans[0]}')
    print(f'dW/dy{zero} = {ans[1]}')
    print('')