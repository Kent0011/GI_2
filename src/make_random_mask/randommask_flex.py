import cv2
import numpy as np
import random
from tqdm import tqdm
import os


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


if __name__ == '__main__':
    
    #================================================================

    SIZE = 1000 # 画像サイズ
    BLOCK = 5 # パターンブロックサイズ
    NUM_OF_IMAGE = 1
    VARIABLE = True # True: 可変, False: 固定
    CUT_SIZE = 20
    FILETYPE = '.bmp'
    FILENAME = f'/Users/kent/Desktop/GI_2/mask/randomimage_size{SIZE}_block{BLOCK}_num{NUM_OF_IMAGE}/'
    
    #================================================================
    size = SIZE
    block = BLOCK
    filename = FILENAME
    
    make_maskset(size=size, filename=filename, num=NUM_OF_IMAGE, block=block)
