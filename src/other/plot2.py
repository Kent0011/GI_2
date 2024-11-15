from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import csv

with open('/Users/kent/Desktop/GI_2/result_text/process_result/5px_200_simulation.txt', 'r', encoding='utf-8') as file:
    data200 = []
    for line in file:
        data200.append(list(map(float,line.split())))
        
with open('/Users/kent/Desktop/GI_2/result_text/process_result/5px_400_simulation.txt', 'r', encoding='utf-8') as file:
    data400 = []
    for i,line in enumerate(file):
        if i%4==0: data400.append(list(map(float,line.split())))

        
data200 = np.array(data200).T
data400 = np.array(data400).T

x = np.array(range(1,1001))
x = x*0.0025
fig, (ax1, ax2) = plt.subplots(1,2)

# --- 左のY軸にプロット ---
ax1.set_xlabel('measurements/pixels (%)')  # X軸のラベル
ax1.set_ylabel('avarage(px)')  # 左のY軸のラベル
ax1.plot(x, data400[0], color = "blue", label='400')
ax1.plot(x, data200[0], color = "red", label='200')

# --- 左のY軸にプロット ---
ax2.set_xlabel('measurements/pixels (%)')  # X軸のラベル
ax2.set_ylabel('var (px^2)')  # 左のY軸のラベル
ax2.plot(x, data400[1], color = "blue", label='400')
ax2.plot(x, data200[1], color = "red", label='200')

# 凡例の表示
ax1.legend()
ax2.legend()

# グラフの表示
plt.show()