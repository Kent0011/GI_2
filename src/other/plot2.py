from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import csv

with open('/Users/kent/Desktop/GI_2/result_text/process_result/5px_data_experiment.txt', 'r', encoding='utf-8') as file:
    data5 = []
    for line in file:
        data5.append(list(map(float,line.split())))
        
with open('/Users/kent/Desktop/GI_2/result_text/process_result/5px_520_simulation.txt', 'r', encoding='utf-8') as file:
    data3 = []
    for line in file:
        data3.append(list(map(float,line.split())))

        
data5 = np.array(data5).T
data3 = np.array(data3).T

x = list(range(1,5001))
x2 = list(range(1,741))
fig, (ax1, ax2) = plt.subplots(1,2)

# --- 左のY軸にプロット ---
ax1.set_xlabel('520')  # X軸のラベル
ax1.set_ylabel('average (px)')  # 左のY軸のラベル
ax1.plot(x, data3[0], color = "blue", label='avarage')
ax1.plot(x, data3[1], color = "red", label='var')

# --- 左のY軸にプロット ---
ax2.set_xlabel('200')  # X軸のラベル
ax2.set_ylabel('average (px)')  # 左のY軸のラベル
ax2.plot(x2, data5[0], color = "blue", label='avarage')
ax2.plot(x2, data5[1], color = "red", label='var')

# 凡例の表示
ax1.legend()
ax2.legend()

# グラフの表示
plt.show()