from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import csv

with open('/Users/kent/Desktop/GI_2/result_text/process_result/5px_data_simulation.txt', 'r', encoding='utf-8') as file:
    data5 = []
    for line in file:
        data5.append(list(map(float,line.split())))
        
with open('/Users/kent/Desktop/GI_2/result_text/process_result/3px_data_simulation.txt', 'r', encoding='utf-8') as file:
    data3 = []
    for line in file:
        data3.append(list(map(float,line.split())))
        
with open('/Users/kent/Desktop/GI_2/result_text/process_result/1px_data_simulation.txt', 'r', encoding='utf-8') as file:
    data1 = []
    for line in file:
        data1.append(list(map(float,line.split())))
        
data5 = np.array(data5).T
data3 = np.array(data3).T
data1 = np.array(data1).T
x = list(range(1,5001))

fig, (ax1, ax2) = plt.subplots(1,2)

# --- 左のY軸にプロット ---
ax1.set_xlabel('number')  # X軸のラベル
ax1.set_ylabel('average (px)')  # 左のY軸のラベル
ax1.plot(x, data1[0], color = "green", label='1px')
ax1.plot(x, data3[0], color = "blue", label='3px')
ax1.plot(x, data5[0], color = "red", label='5px')

# --- 右のY軸を追加してプロット ---
ax2.set_xlabel('number')
ax2.set_ylabel('var (px^2)')  # 右のY軸のラベル
ax2.plot(x, data1[1], color = "green", label='1px')
ax2.plot(x, data3[1],color = "blue", label='3px')
ax2.plot(x, data5[1],color = "red", label='5px')

# 凡例の表示
ax1.legend()
ax2.legend()

# グラフの表示
plt.show()