import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os
import json
import requests

# ==============================================================

BLOCK = 5
NUM = 5000
DOTS_POSITION_FILE = "/Users/kent/Desktop/GI_2/image/5*5dots_random.scv" 
EXPERIMENT_RESULT_FILE = f"/Users/kent/Desktop/GI_2/result_text/experiment_result/block{BLOCK}_5000.txt"
MASK_NAME = f"/Users/kent/Desktop/GI_2/mask/randomimage_size200_block{BLOCK}_num{NUM}/"
SIZE = 200
DOTS_NUM = 5
RESULT_TEXT_FILE = f'/Users/kent/Desktop/GI_2/result_text/process_result/{BLOCK}px_data.txt'
RESULT_FIG_FILE = f'/Users/kent/Desktop/Kenkyu/plot/{BLOCK}px_data_test.png'


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


def calcCGI(measurment_num, size, b_path, mask_folder):
    I = np.zeros((size, size))
    BI = np.zeros((size, size))

    T = np.zeros((size, size))
    T1 = np.zeros((size, size))
    T2 = np.zeros((size, size))

    f = open(b_path, 'r')
    B = 0
    height = size
    width = size

    # Bの正規化
    imglist = []
    min = 0

    f = open(b_path, 'r')
    for i in tqdm(range(measurment_num)):

        mask = cv2.imread(mask_folder + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE)
        mask = 255 - mask
        mask = mask / 255

        Bvalue = f.readline()
        light_value = float(Bvalue)
        B += light_value
        square_x = (width - size) // 2
        square_y = (height - size) // 2

        mask_sl = mask[square_y:square_y+size, square_x:square_x+size]
   
        I += mask_sl
        BI += (mask_sl*light_value)
        
                
        tmp = BI/(i+1) - ((B/(i+1)) * (I/(i+1)))
        min = np.min(tmp)
        max = np.max(tmp)

        T2 = (tmp - min)/(max - min)

        T2a = np.clip(T2 * 255, 0, 255).astype(np.uint8)
        imglist.append(T2a)

    f.close()
    return imglist
    
    
if __name__ == '__main__':
    
    column_names = ['y','x']
    dots_5000 = pd.read_csv(DOTS_POSITION_FILE, header=None, names=column_names)
    
    ans = []
    images = calcCGI(
        measurment_num=NUM,
        size = SIZE,
        b_path = EXPERIMENT_RESULT_FILE,
        mask_folder=MASK_NAME)
    
    for image in tqdm(images):
        
        dots = read_image(img=image, dots_vertical=DOTS_NUM, dots_horizontal=DOTS_NUM, title='test')
        
        data = {
            'y': [dot[0] for dot in dots],
            'x': [dot[1] for dot in dots]
        }
        dots = pd.DataFrame(data)
        
        distances = np.sqrt((dots_5000['x'] - dots['x']) ** 2 + (dots_5000['y'] - dots['y']) ** 2)
        average_distance = distances.mean()
        ans.append([average_distance,np.var(distances, ddof=1)])
        
    with open(RESULT_TEXT_FILE, 'w') as file:
        for number in ans:
            file.write(f"{number[0]} {number[1]}\n")
    
    plt.plot(ans)
    plt.savefig(RESULT_FIG_FILE)
    plt.show()
    
    def convert_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    images = list(map(convert_to_rgb,images))
    
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])  # 最初の画像を表示

    # アニメーション用の更新関数
    def update(frame):
        im.set_array(images[frame])
        return [im]

    # アニメーションの作成
    ani = animation.FuncAnimation(fig, update, frames=len(images), repeat=True)

    # アニメーションを表示
    plt.axis('off')  # 軸を非表示にする
    plt.show()