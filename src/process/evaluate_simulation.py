import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import json
import requests

# ==============================================================

BLOCK = 5
NUM = 40
SIZE = 40
DOTS_POSITION_FILE = "/Users/kent/Desktop/GI_2/image/test40_dots.csv"
OBJECT_NAME = "/Users/kent/Desktop/GI_2/image/test40.bmp"
MASK_NAME = f"/Users/kent/Desktop/GI_2/mask/randomimage_size{SIZE}_block{BLOCK}_num40/"
DOTS_NUM = 1
RESULT_TEXT_FILE = f'/Users/kent/Desktop/GI_2/result_text/process_result/{BLOCK}px_40_simulation.txt'
RESULT_FIG_FILE = f'/Users/kent/Desktop/Kenkyu/plot/{BLOCK}px_40_simulation.png'


# ==============================================================


def serach_brightest_grid(img: list, size: int, vert_0:int, hrzn_0:int) -> list:
    
    csum = np.zeros((len(img)+1,len(img[0])+1))
    
    for i,x in enumerate(img):
        for j,brightness in enumerate(x):
            csum[i+1][j+1] = np.mean(brightness)+csum[i][j+1]+csum[i+1][j]-csum[i][j]
            
    brightness = 0
    destination = [0,0]
    for i in range(size,len(csum)):
        for j in range(size,len(csum[0])):
            if brightness <= (tmp:=csum[i][j] - csum[i-size][j] - csum[i][j-size] + csum[i-size][j-size]):
                brightness = tmp
                destination = [i-1+vert_0,j-1+hrzn_0]
                
    return destination




def serach_centroid(img: list, vert_0:int, hrzn_0:int) -> list:
    
    power = np.zeros((len(img),len(img[0])))
    
    for i,x in enumerate(img):
        for j,brightness in enumerate(x):
            power[i][j] = np.mean(brightness)

    brightness_sum = sum(sum(x) for x in power)
    
    x_g, y_g = 0,0
    
    for i,x in enumerate(power):
        for j,brightness in enumerate(x):
            x_g+=i*brightness
            y_g+=j*brightness
    if brightness_sum!=0:
        x_g = x_g/brightness_sum
        y_g = y_g/brightness_sum
    
    return [y_g+vert_0, x_g+hrzn_0]




def read_image(img: list, dots_vertical: int, dots_horizontal: int, title:str) -> list:
    
    height = len(img)//dots_vertical
    width = len(img[0])//dots_horizontal
    dot_position = []
        
    # 中心位置検出
    for i in range(dots_vertical):
        for j in range(dots_horizontal):
            tmp = serach_brightest_grid(img[height*i:height*(i+1), width*j:width*(j+1)], 8, height*i, width*j)
            dot_position.append(tmp)
    
    ans = []        

    # 重心計算
    for dot in dot_position:
        grid = img[dot[0]-7:dot[0]+1, dot[1]-7:dot[1]+1]
        cent = serach_centroid(grid,dot[0]-7, dot[1]-7)
        ans.append(cent)
            
    return ans


def calcCGI_simulation(num, maskname, size, objname):
    B = 0
    I = np.zeros((size, size))
    BI = np.zeros((size, size))
    T2 = np.zeros((size, size))
    imglist = []

    img = cv2.imread(objname)
    while(img is None):
        img = cv2.imread(objname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    
    for i in tqdm(range(num)):  
        mask = cv2.cvtColor(cv2.imread(maskname + str(i) + '.bmp'), cv2.COLOR_BGR2GRAY)
        mask = mask / 255

        In = np.multiply(img,mask) #要素同士をかけ合わせてる

        light_value = 0
        # 画素値から強度を計算する
        light_value = np.sum(In)
 
        B += light_value
        I += mask
        BI += (mask * light_value)
        
        # 再構成画像出力
        tmp = BI/(i+1) - (B/(i+1))*(I/(i+1))
        max = np.max(tmp)
        min = np.min(tmp)

        T2 = (tmp - min) / (max - min)

        #0-1から255階調に変更
        T2 = np.clip(T2 * 255, 0, 255).astype(np.uint8)
        
        imglist.append(T2)

    return imglist
    
    
if __name__ == '__main__':
    
    column_names = ['y','x']
    dots_ans = pd.read_csv(DOTS_POSITION_FILE, header=None, names=column_names)
    
    ans = []
    images = calcCGI_simulation(
        num=NUM,
        objname=OBJECT_NAME,
        maskname=MASK_NAME,
        size = SIZE)
    
    for image in tqdm(images):
        
        dots = read_image(img=image, dots_vertical=DOTS_NUM, dots_horizontal=DOTS_NUM, title='test')
        
        data = {
            'y': [dot[0] for dot in dots],
            'x': [dot[1] for dot in dots]
        }
        dots = pd.DataFrame(data)
        
        distances = np.sqrt((dots_ans['x'] - dots['x']) ** 2 + (dots_ans['y'] - dots['y']) ** 2)
        average_distance = [distances.mean(), np.var(distances, ddof=1)]
        ans.append(average_distance)
        
    with open(RESULT_TEXT_FILE, 'w') as file:
        for number in ans:
            file.write(f"{number[0]} {number[1]}\n")
        
    plt.plot(ans)
    plt.savefig(RESULT_FIG_FILE)
    plt.show()
    