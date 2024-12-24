from fileinput import filename
import cv2
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import random


def makeRandomMask(size: int, dirname: str, num: int, block: int, filetype='.bmp', cut_size=20, variable=True) -> None:
    
    rand_size = size
    
    if variable == True: rand_size += cut_size
        
    width, height = rand_size, rand_size
    
    # filenameが存在しないとき作成
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # ランダムパターンを作成
    Random_img = np.zeros((height,width), dtype=np.uint8)
    for i in range(0, height, block):
        for j in range(0, width, block):
            if random.randint(1,100) <= 50:
                Random_img[i:i+block, j:j+block] = 0
            else:
                Random_img[i:i+block, j:j+block] = 255
    
    if variable == True:
        # パターンを規定サイズにカット
        r = random.randint(0, cut_size)
        img = Random_img[r:r+size, r:r+size]

    maskName = dirname + str(num) + filetype
    cv2.imwrite(maskName, img)
    

def make_maskset(size: int, filename: str, num: int, block: int, filetype='.bmp', cut_size=20, variable=True) -> None:
    
    for i in tqdm(range(num)):
        makeRandomMask(size, filename, i, block, filetype, cut_size, variable)


def simulate(num: int, objname: str, maskname: str, size: int, resultfolder: str, resultname: str) -> list:
    B = 0
    I = np.zeros((size, size))
    BI = np.zeros((size, size))
    T2 = np.zeros((size, size))

    img = cv2.imread(objname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    for i in range(num):
        mask = cv2.cvtColor(cv2.imread(
            maskname + str(i) + '.bmp'), cv2.COLOR_BGR2GRAY)
        mask = mask / 255

        In = np.multiply(img, mask)  # 要素同士をかけ合わせてる

        light_value = 0
        # 画素値から強度を計算する
        light_value = np.sum(In)

        # Bはスカラー、
        B = B + light_value / num

        # IとBIの計算
        I = I + mask / num
        BI = BI + mask * light_value / num

    # 再構成画像出力
    tmp = BI - B*I
    max = np.max(tmp)
    min = np.min(tmp)

    T2 = (tmp - min) / (max - min)

    # 0-1から255階調に変更
    T2 = np.clip(T2 * 255, 0, 255).astype(np.uint8)

    plt.imshow(T2)
    plt.show()

    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    resultfile = resultfolder + resultname

    if cv2.imwrite(resultfile, T2):
        print('simuration succed')
    
    return T2


def search_brightest_grid(img: list, size: int, vert_0:int, hrzn_0:int) -> list:
    
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


def search_centroid(img: list, vert_0:int, hrzn_0:int) -> list:
    
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


def search_dot(img: list, dots_vertical: int, dots_horizontal: int, grid_size: int = 7) -> list:
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    height = len(img)//dots_vertical
    width = len(img[0])//dots_horizontal
    dot_position = []
        
    # 中心位置検出
    for i in range(dots_vertical):
        for j in range(dots_horizontal):
            tmp = search_brightest_grid(img[height*i:height*(i+1), width*j:width*(j+1)], grid_size, height*i, width*j)
            dot_position.append(tmp)
            img[int(tmp[0])][int(tmp[1])] = [0,0,255]
            img[int(tmp[0]-grid_size+1)][int(tmp[1])] = [0,0,255]
            img[int(tmp[0])][int(tmp[1]-grid_size+1)] = [0,0,255]
            img[int(tmp[0]-grid_size+1)][int(tmp[1]-grid_size+1)] = [0,0,255]
    
    print(dot_position)
    
    # マス目描画
    for i, vert in enumerate(img):
        for j, hrzn in enumerate(vert):
            if i%height == 0 or j%width == 0:
                img[i][j] = [200,200,200]
            
    # 重心計算
    for dot in dot_position:
        grid = img[dot[0]-grid_size+1:dot[0]+1, dot[1]-grid_size+1:dot[1]+1]
        cent = search_centroid(grid,dot[0]-grid_size+1, dot[1]-grid_size+1)
        print(*cent)
        img[round(cent[0])][round(cent[1])] = [255,0,0]
    
    print("position: (i, j)")
    for dot in dot_position:
        print(f"({dot[0]}, {dot[1]})")
            
    # 画像表示
    plt.imshow(img)
    plt.show()
    
    return img


if __name__ == "__main__":
    
    # =====================================================
    NUMBER = 5000
    OBJECT_PASS = "/Users/kent/Desktop/GI_2/src/zernike/dots.bmp"
    MASK_PASS = "/Users/kent/Desktop/GI_2/mask/randomimage_size200_block5_num5000/"
    SIZE = 200
    RESULT_PASS = "/Users/kent/Desktop/GI_2/src/zernike/"
    RESULTNAME = "GI_simuration_test.bmp"
    # =====================================================
    
    search_dot(
        simulate(NUMBER, OBJECT_PASS, MASK_PASS, SIZE, RESULT_PASS, RESULTNAME),
        dots_vertical=5,
        dots_horizontal=5)
