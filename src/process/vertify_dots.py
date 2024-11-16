import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os


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


def read_image(image_pass: str, dots_vertical: int, dots_horizontal: int, size: int) -> list:
    
    img = cv2.imread(image_pass)
    
    # img = img[:190, :190] #位置調整
    
    height = len(img)//dots_vertical
    width = len(img[0])//dots_horizontal
    dot_position = []
        
    # 中心位置検出
    for i in range(dots_vertical):
        for j in range(dots_horizontal):
            tmp = serach_brightest_grid(img[height*i:height*(i+1), width*j:width*(j+1)], size, height*i, width*j)
            dot_position.append(tmp)
            img[int(tmp[0])][int(tmp[1])] = [0,0,255]
            img[int(tmp[0]-size+1)][int(tmp[1])] = [0,0,255]
            img[int(tmp[0])][int(tmp[1]-size+1)] = [0,0,255]
            img[int(tmp[0]-size+1)][int(tmp[1]-size+1)] = [0,0,255]
    
    # マス目描画
    for i, vert in enumerate(img):
        for j, hrzn in enumerate(vert):
            if i%height == 0 or j%width == 0:
                img[i][j] = [200,200,200]
            
    # 重心計算
    for dot in dot_position:
        grid = img[dot[0]-size+1:dot[0]+1, dot[1]-size+1:dot[1]+1]
        cent = serach_centroid(grid,dot[0]-size+1, dot[1]-size+1)
        print(*cent)
        img[round(cent[0])][round(cent[1])] = [255,0,0]
            
    # 画像表示
    plt.imshow(img)
    plt.savefig(f'/Users/kent/Desktop/GI_2/result_image/452_dots')
    plt.show()
    
    return img

if __name__ == '__main__':
    
    dots = 5
    
    read_image(
    image_pass = '/Users/kent/Desktop/GI_2/result_image/GI_simuration_test452.bmp',
    dots_vertical = dots,
    dots_horizontal = dots,
    size = 7)
    
    
    